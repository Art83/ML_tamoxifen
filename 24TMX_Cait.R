
library(factoextra)
library(FactoMineR)


#data
dat <- read.csv('LT.csv', row.names = 1)




# Dataset doesn't contain NAs but is not normalized

data <- as.data.frame(scale(dat[,-1]))
data$Group <- as.factor(dat$Group)


colnames(data)[1:3] <- c("Exploration Proportion", "Time Moved", "Time Unmoved")







# PCA
res.pca <- PCA(data[,-10], graph = F)


png("PCA.png", units = "in", width = 6, height = 5, res = 300)
fviz_pca_biplot(res.pca, 
                # Fill individuals by groups
                geom.ind = "point",
                pointshape = 21,
                pointsize = 2.5,
                fill.ind = data$Group,
                col.var = "black",
                #col.ind = data$Group,
                # Color variable by groups
                legend.title = list(fill = "Groups"),
                repel = TRUE,        # Avoid label overplotting
                axes = c(1,2)
)+
  ggpubr::fill_palette("jco")+      # Indiviual fill color
  ggpubr::color_palette("npg")      # Variable colors
dev.off()

fviz_contrib(res.pca, choice = 'var', axes = 1:2, top = 10)


# PC1
cr <- prcomp(data[,-10])
cr$x[,1]



new_dat <- data.frame(Group = as.character(data[,10]), value = as.numeric(cr$x[,1]))


ggplot(new_dat, aes(x = as.factor(Group), y = as.numeric(value), color = Group))+
  geom_boxplot()

fit <- aov(as.numeric(value) ~ Group, new_dat)
summary(fit)
TukeyHSD(fit)

cohen.d(as.numeric(new_dat$value[new_dat$Group == 'SI-Veh']), as.numeric(new_dat$value[new_dat$Group == 'Sh-TMX']))



