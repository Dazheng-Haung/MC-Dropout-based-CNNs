## load packages
library(gstat)      # for geostats
library(dplyr)      # for data manipulation
library(magrittr)   # for functional programming
library(ggplot2)    # for nice plots
library(readxl)     # for reading XLSx files
library(compositions)     # for CoDA
library(sp)     # for spatial objects
library(gmGeostats)
library(sf)
library(tidyverse)
library(caret)
library(raster)
library(openxlsx)




# for reading XLSX filesgeoch(LOG transformation)
geochemistry_df <- read_excel("geochemistry4_clean_R.xlsx")
geochemistry_xy = geochemistry_df %>% dplyr::select(x, y) %>% SpatialPoints()
geochemistry_spdf = geochemistry_df %>%
  dplyr::select(As:Sb) %>%
  log10() %>%
  SpatialPointsDataFrame(coords = geochemistry_xy)
geochemistry_compo = geochemistry_df %>%
  dplyr::select(As:Sb) %>%
  log10()

geochemistry.coords = as.matrix(geochemistry_df[,c('x','y')])



# create grid
xymin = sapply(geochemistry_df[,1:2], min)
xymax = sapply(geochemistry_df[,1:2], max)
xdens = seq(from=xymin[1], to=xymax[1], by=1000)
ydens = seq(from=xymin[2], to=xymax[2], by=1000)
#geochemistry.grid.fine = expand.grid(x=xdens, y=ydens)
x0 = c(xymin[1], xymin[2])
names(x0) = colnames(geochemistry.coords)
Dx = c(1000,1000)
nx = c(xymax[1]-xymin[1], xymax[2]-xymin[2])/Dx+1
geocchemistry.gt = GridTopology(x0, Dx, nx)
geochemistry.grid.fine = SpatialGrid(geocchemistry.gt)

#### Masking 
geochemistry.mask.fine = constructMask(geochemistry.grid.fine, maxval=2001,
                                       x= cbind(geochemistry.coords,geochemistry_compo))
geochemistry.grid.fine.masked = setMask(geochemistry.grid.fine,  geochemistry.mask.fine)


##
####cross variograms using log data
###
geochemistry_gg = make.gmMultivariateGaussianSpatialModel(data = geochemistry_spdf, nmax=200)
geochemistry_vg = variogram(geochemistry_gg,  cutoff=25000, width=750)
plot(geochemistry_vg)
#Fit 
geochemistry_vgm = vgm(range=6000, model="Sph", psill=0.3, nugget = 0.2)
geochemistry_vgm = vgm(range=25000, model="Sph", psill=0.3, add.to= geochemistry_vgm)

geochemistry_gs = fit_lmc(v = geochemistry_vg, model=geochemistry_vgm, 
                          g=geochemistry_gg, correct.diagonal = 1.0001)
plot(geochemistry_vg, model=geochemistry_gs)


##
#### SGS using log data
##
geochemistry.sim = predict(geochemistry_gs, newdata = geochemistry.grid.fine, 
                           nsim=200, debug.level=-1)
colnames(geochemistry.sim)
dim(geochemistry.sim)
summary(geochemistry.sim)
summary(geochemistry_spdf)
geochemistry.sim.save = data.frame(geochemistry.sim)
geochemistry.sim.save_mask = data.frame(geochemistry.sim)
summary(geochemistry.sim.save)

# 获取所有以 "As.sim"、"Au.sim"、"Hg.sim" 和 "Sb.sim" 开头的列名
sim_columns <- grep("^(As\\.sim|Au\\.sim|Hg\\.sim|Sb\\.sim)", colnames(geochemistry.sim.save), value = TRUE)


for (sim_col in sim_columns) {
  
  geochemistry_matrix <- geochemistry.sim.save[, c("x", "y", sim_col)]
  
  geochemistry_matrix_spread <- xtabs(get(sim_col) ~ y + x, data = geochemistry_matrix)

  file_name <- paste0(sim_col, "_matrix.txt")
  write.table(geochemistry_matrix_spread, file = file_name, sep = "\t", row.names = FALSE, col.names = FALSE)
  print(paste0(file_name, " 已成功保存"))
}
