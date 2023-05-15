# Title     : TODO
# Objective : TODO
# Created by: stefa
# Created on: 30/05/2021

calculate_pcnm <- function(matrix, distance=NULL) {
  library(vegan)
  distance <-ifelse(is.null(distance), median(matrix), distance)
  pcnm1 <- pcnm(matrix, distance)
  return(pcnm1)
}