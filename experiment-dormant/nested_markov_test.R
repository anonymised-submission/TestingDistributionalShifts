# the source code is from https://github.com/rje42/ADMGs2
source('R/fitADMG.R')
library(MixedGraphs)
library(hash)

setwd("R")
files.sources = list.files()
sapply(files.sources, source)
setwd('../')


nmm_test <- function(dat) {
graph1 <- graphCr("1->2->3->4","2<->4")
graph2 <- graphCr("1->2->3->4","2<->4","1->4")

ll1 <- fitADMG(dat, graph1, r=TRUE)
ll2 <- fitADMG(dat, graph2, r=TRUE)

teststat <- -2 * (ll1$ll-ll2$ll)
df <- ll2$p-ll1$p
p.value <- pchisq(teststat, df = df, lower.tail = FALSE)
p.value
}