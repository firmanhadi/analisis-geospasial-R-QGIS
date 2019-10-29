# Code modified from https://valentinitnelav.github.io/satellite-image-classification-r/
# Packages for spatial data processing & visualization
library(rgdal)
library(gdalUtils)
library(raster)
library(sf)
library(sp)
library(RStoolbox)
library(getSpatialData)
library(rasterVis)
library(mapview)

library(RColorBrewer)
library(plotly)
library(grDevices)

# Machine learning packages
library(caret)
library(randomForest)
library(ranger)
library(MLmetrics)
library(nnet)
library(NeuralNetTools)
library(LiblineaR)

# Packages for general data processing and parallel computation
library(data.table)
library(dplyr)
library(stringr)
library(doParallel)
library(snow)
library(parallel)

# set the temporary folder for raster package operations
rasterOptions(tmpdir = "./cache/temp")


#AOI

aoi <- matrix(data = c(22.85, 45.93,  # Upper left corner
                       22.95, 45.93,  # Upper right corner
                       22.95, 45.85,  # Bottom right corner
                       22.85, 45.85,  # Bottom left corner
                       22.85, 45.93), # Upper left corner - closure
              ncol = 2, byrow = TRUE)
set_aoi(aoi)

view_aoi()

#Make Raster list
ras <- list.files(path="TIF", pattern = ".tif$", full.names = TRUE)

#Stack Raster list
stack_ras <- raster::stack(ras)

#Make RasterBrick
brick_ras <- brick(stack_ras)

#Normalization
brick_ras_norm <- normImage(brick_ras)

#Copy names of RasterBrick to Normalized Raster Brick
names(brick_ras_norm) <- names(brick_ras)

#Read Shapefile
poly <- rgdal::readOGR(dsn   = "./data/train_polys", 
                       layer = "train_polys", 
                       stringsAsFactors = FALSE)

# Need to have a numeric id for each class - helps with rasterization later on.
poly@data$id <- as.integer(factor(poly@data$class))
setDT(poly@data)

# Prepare colors for each class.
cls_dt <- unique(poly@data) %>% 
  arrange(id) %>% 
  mutate(hex = c(non-peatland  = "#ff7f00",
                 peatland = "#e41a1c"
  ))


view_aoi(color = "#a1d99b") + 
  mapView(poly, zcol = "class", col.regions = cls_dt$hex)

poly_utm <- sp::spTransform(poly, CRSobj = brick_ras$TWI90@crs)

# Create raster template
template_rst <- raster(extent(brick_ras$TWI90), # band B2 has resolution 10 m
                       resolution = 30,
                       crs = projection(brick_ras$TWI90))

poly_utm_rst <- rasterize(poly_utm, template_rst, field = 'id')

poly_dt <- as.data.table(rasterToPoints(poly_utm_rst))
setnames(poly_dt, old = "layer", new = "id_cls")

points <- SpatialPointsDataFrame(coords = poly_dt[, .(x, y)],
                                 data = poly_dt,
                                 proj4string = poly_utm_rst@crs)

dt <- brick_ras_norm %>% 
  extract(y = points) %>% 
  as.data.table %>% 
  .[, id_cls := points@data$id_cls] %>%  # add the class names to each row
  merge(y = unique(poly@data), by.x = "id_cls", by.y = "id", all = TRUE, sort = FALSE) %>% 
  .[, id_cls := NULL] %>% # this column is extra now, delete it
  .[, class := factor(class)]

# View the first 6 rows
head(dt)

dt %>% 
  select(-"class") %>% 
  melt(measure.vars = names(.)) %>% 
  ggplot() +
  geom_histogram(aes(value)) +
  geom_vline(xintercept = 0, color = "gray70") +
  facet_wrap(facets = vars(variable), ncol = 3)

set.seed(321)
# A stratified random split of the data
idx_train <- createDataPartition(dt$class,
                                 p = 0.7, # percentage of data as training
                                 list = FALSE)
dt_train <- dt[idx_train]
dt_test <- dt[-idx_train]

table(dt_train$class)

table(dt_test$class)

# create cross-validation folds (splits the data into n random groups)
n_folds <- 10
set.seed(321)
folds <- createFolds(1:nrow(dt_train), k = n_folds)
# Set the seed at each resampling iteration. Useful when running CV in parallel.
seeds <- vector(mode = "list", length = n_folds + 1) # +1 for the final model
for(i in 1:n_folds) seeds[[i]] <- sample.int(1000, n_folds)
seeds[n_folds + 1] <- sample.int(1000, 1) # seed for the final model
ctrl <- trainControl(summaryFunction = multiClassSummary,
                     method = "cv",
                     number = n_folds,
                     search = "grid",
                     classProbs = TRUE, # not implemented for SVM; will just get a warning
                     savePredictions = TRUE,
                     index = folds,
                     seeds = seeds)


cl <- makeCluster(3/4 * detectCores())
registerDoParallel(cl)
model_rf <- caret::train(class ~ . , method = "rf", data = dt_train,
                         importance = TRUE, # passed to randomForest()
                         # run CV process in parallel;
                         # see https://stackoverflow.com/a/44774591/5193830
                         allowParallel = TRUE,
                         tuneGrid = data.frame(mtry = c(2, 3, 4, 5, 8)),
                         trControl = ctrl)
stopCluster(cl); remove(cl)
# Unregister the doParallel cluster so that we can use sequential operations
# if needed; details at https://stackoverflow.com/a/25110203/5193830
registerDoSEQ()
saveRDS(model_rf, file = "./cache/model_rf.rds")
model_rf$times$everything # total computation time

plot(model_rf) # tuning results
