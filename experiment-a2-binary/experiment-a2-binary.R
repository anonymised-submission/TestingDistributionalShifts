setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(RColorBrewer)
library(tikzDevice)

use.tikz <- F

# Load
df <- read_delim("experiment-a2-binary.csv", delim=",", col_types = cols(n="f", Power="f", SamplingScheme="f")) %>%
  transform(level = addNA(factor(RejectRate < 0.05))) %>%
  subset(SamplingScheme == "False")

method.names <- c("False" = "NO-REPL", "True" = "REPL")
df$SamplingScheme <- factor(df$SamplingScheme, levels=names(method.names), labels = method.names)


# Setup tikz
path = "experiment-a2-binary"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 4, height =4)}

levels(df$level)

p <- df %>% 
  ggplot(aes(x=n, y=Power, fill=RejectRate, color = level, width=0.85, height=0.85)) + 
  geom_tile(size=0.8)+
  scale_fill_gradientn(colours=rev(brewer.pal(10,"Spectral")))+ 
  scale_x_discrete(breaks=c("100", "1000", "10000", "100000", "1000000")) +
  labs(x="Sample size n",
       y = "Exponent a, $m = n^a$",
       color = "Rate below 5\\%",
       fill = "Rejection rate") +
  coord_fixed() + facet_wrap(~SamplingScheme)
print(p)

if(use.tikz){
  dev.off()
  print(p)
  
  # Handle scale in right bar
  file <- readLines(paste0(path, ".tex"))
  y <- gsub("_ras", "-ras", file)
  y <- gsub("experiment-a2-binary-ras1", "experiment-a2-binary-ras1", y)
  cat(y, file=paste0(path,".tex"), sep="\n")
  file.rename(paste0(path,"_ras1.png"), 
              paste0(path,"-ras1.png"))
  
  lines <- readLines(con=paste0(path, ".tex"))
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=paste0(path, ".tex"))
  ggsave(paste0(path, ".pdf"))
  
}


