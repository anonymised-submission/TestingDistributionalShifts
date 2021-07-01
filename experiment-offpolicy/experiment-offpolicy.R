# Automatically set workdir to source folder (only works in RStudio)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(gridExtra)
library(tidyverse)
library(tikzDevice)
library(RColorBrewer)
f <- function(pal) brewer.pal(brewer.pal.info[pal, "maxcolors"], pal)
colors <- f("Dark2")


use.tikz <- F
path = "experiment-offpolicy"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 5.5, height = 2)}



grid_arrange_shared_legend <- function(..., ncol = length(list(...)), nrow = 1, position = c("bottom", "right")) {
  
  plots <- list(...)
  position <- match.arg(position)
  g <- ggplotGrob(plots[[2]] + theme(legend.position = position))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  lwidth <- sum(legend$width)
  gl <- lapply(plots, function(x) x + theme(legend.position="none"))
  gl <- c(gl, ncol = ncol, nrow = nrow)
  
  combined <- switch(position,
                     "bottom" = arrangeGrob(do.call(arrangeGrob, gl),
                                            legend,
                                            ncol = 1,
                                            heights = unit.c(unit(1, "npc") - lheight, lheight)),
                     "right" = arrangeGrob(do.call(arrangeGrob, gl),
                                           legend,
                                           ncol = 2,
                                           widths = unit.c(unit(1, "npc") - lwidth, lwidth)))
  
  grid.newpage()
  grid.draw(combined)
  
  # return gtable invisibly
  invisible(combined)
  
}

# Load
df <- read_delim("experiment-offpolicy.csv", delim=",")
df$Policy_Effect = (df$Policy_Effect - min(df$Policy_Effect))/(max(df$Policy_Effect) - min(df$Policy_Effect))

# Set confidence level
conf.level = 0.05
# colors = scales::hue_pal()(3)
first_col = colors[1]
sec_col = colors[2]
third_col = colors[3]
sa_factor = 1.

mytheme <- list(theme_bw(), 
                scale_color_manual(values = c(sec_col, first_col, third_col)),
                theme(
                  legend.text=element_text(size=7),
                  legend.position="top",
                  axis.title.y.right = element_text(color = sec_col),
                  plot.margin = margin(0, 5, 0, 5),
                  axis.title.y = element_text(margin = margin(t = 0, r = -5, b = 0, l = -10))
                )
                )



p1 <- df %>%
  ggplot(aes(x=Policy_Effect)) +
  geom_ribbon(aes(colour=NULL,ymin=0,ymax=conf.level, fill="confint"), alpha=0.8, fill="grey70", show.legend = F) +
  geom_point(aes(y=alpha, color='Rejection Rate'), size=1.) +
  geom_point(aes(y=mean_effect*sa_factor, color='Estimated Exp. Reward'), size=1.) +
  geom_line(aes(y=alpha, color='Rejection Rate')) +
  geom_line(aes(y=mean_effect*sa_factor, color='Estimated Exp. Reward')) +
  geom_ribbon(aes(ymax=Upper, ymin=Lower), size=0.3,alpha=.2, color=first_col, fill=first_col) +
  geom_ribbon(aes(ymax=Upper_me*sa_factor, ymin=Lower_me*sa_factor), size=0.3,alpha=.2, color=sec_col, fill=sec_col)+
  geom_hline(aes(lty="$5\\%$ level", yintercept=conf.level), size=0.1,  show.legend = T) +
  labs(y="Rejection rate", x="Policy strength ($\\delta$)", color='')+
  scale_y_continuous(
    labels = function(z){paste0(100*z, "\\%")},
    # Features of the first axis
    name = "Rejection Rate",
    # Add a second axis and specify its features
    sec.axis = sec_axis(~./sa_factor, name="Exp. Reward under $p^*(a|z)$")
  ) +
  scale_linetype_manual(values = c("22"), breaks = "$5\\%$ level", name=NULL) +
  mytheme

# Load
df <- read_delim("experiment-offpolicy-MMD.csv", delim=",")
df$Policy_Effect = (df$Policy_Effect - min(df$Policy_Effect))/(max(df$Policy_Effect) - min(df$Policy_Effect))


p2 <- df %>%
  ggplot(aes(x=Policy_Effect)) +
  geom_ribbon(aes(colour=NULL,ymin=0,ymax=conf.level, fill="confint"), alpha=0.8, fill="grey70", show.legend = F) +
  geom_point(aes(y=alpha, color='Reject. rate Mann-Whitney'), size=1.) +
  geom_point(aes(y=alpha_mmd, color='Reject. rate MMD'), size=1.) +
  geom_point(aes(y=mean_effect*sa_factor, color='Expected reward'), size=1.) +
  geom_line(aes(y=alpha, color='Reject. rate Mann-Whitney')) +
  geom_line(aes(y=alpha_mmd, color='Reject. rate MMD')) +
  geom_line(aes(y=mean_effect*sa_factor, color='Expected reward')) +
  geom_ribbon(aes(ymax=Upper, ymin=Lower), size=0.3,alpha=.2, color=first_col, fill=first_col) +
  geom_ribbon(aes(ymax=Upper_mmd, ymin=Lower_mmd), size=0.3,alpha=.2, color=third_col, fill=third_col)+
  geom_ribbon(aes(ymax=Upper_me*sa_factor, ymin=Lower_me*sa_factor), size=0.3,alpha=.2, color=sec_col, fill=sec_col)+
  geom_hline(aes(lty="5\\% level", yintercept=conf.level), size=0.1,  show.legend = T) +
  labs(y="Rejection rate (R.R.)", x="Policy strength ($\\delta$)", color='')+
  scale_y_continuous( labels = function(z){paste0(100*z, "\\%")},
                      # Features of the first axis
                      name = "Rejection Rate",
                      # Add a second axis and specify its features
                      sec.axis = sec_axis(~./sa_factor, name="Difference in Exp. Reward")
  ) +
  scale_linetype_manual(values = c("22"), breaks = "5\\% level", name=NULL) +
  mytheme

# Load
df <- read_delim("experiment-offpolicy-var.csv", delim=",")
df$Policy_Effect = (df$Policy_Effect - min(df$Policy_Effect))/(max(df$Policy_Effect) - min(df$Policy_Effect))


p3 <- df %>%
  ggplot(aes(x=Policy_Effect)) +
  geom_ribbon(aes(colour=NULL,ymin=0,ymax=conf.level, fill="confint"), alpha=0.8, fill="grey70", show.legend = F) +
  geom_point(aes(y=alpha, color='Reject. rate Mann-Whitney'), size=1.) +
  geom_point(aes(y=alpha_mmd, color='Reject. rate MMD'), size=1.) +
  geom_point(aes(y=mean_effect*sa_factor, color='Expected reward'), size=1.) +
  geom_point(aes(x=0.5, y=1), show.legend=F, alpha=0) + # This point is a hack to get the ylims right
  geom_line(aes(y=alpha, color='Reject. rate Mann-Whitney')) +
  geom_line(aes(y=alpha_mmd, color='Reject. rate MMD')) +
  geom_line(aes(y=mean_effect*sa_factor, color='Expected reward')) +
  geom_ribbon(aes(ymax=Upper, ymin=Lower), size=0.3,alpha=.2, color=first_col, fill=first_col) +
  geom_ribbon(aes(ymax=Upper_mmd, ymin=Lower_mmd), size=0.3,alpha=.2, color=third_col, fill=third_col)+
  geom_ribbon(aes(ymax=Upper_me*sa_factor, ymin=Lower_me*sa_factor), size=0.3,alpha=.2, color=sec_col, fill=sec_col)+
  geom_hline(aes(lty="5\\% level", yintercept=conf.level), size=0.1,  show.legend = T) +
  labs(y="Rejection rate (R.R.)", x="Policy strength ($\\delta'$)", color='')+
  scale_y_continuous( labels = function(z){paste0(100*z, "\\%")},
                      # Features of the first axis
                      name = "Rejection Rate",
                      # Add a second axis and specify its features
                      sec.axis = sec_axis(~./sa_factor, name="Difference in Exp. Reward")
  ) +
  scale_linetype_manual(values = c("22"), breaks = "5\\% level", name=NULL) +
  mytheme

grid_arrange_shared_legend(p1, p2, p3)
if(use.tikz){
  dev.off()
  grid_arrange_shared_legend(p1, p2, p3)
  lines <- readLines(con=paste0(path, ".tex"))
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=paste0(path, ".tex"))
  ggsave(paste0(path, ".pdf"), width = 7, height = 2.8)
  
}



