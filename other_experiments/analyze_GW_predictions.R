library(tidyverse)

# Set ggplot styling
cai_theme<- theme_gray(base_size = 18) + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5), 
        plot.subtitle= element_text(hjust=0.5), 
        legend.text = element_text(size=12), 
        legend.title = element_text(size = 12))
theme_set(cai_theme)

#####
#
# EXPLORATIONS AND VISUALIZATIONS
#
####3

# Load data and compute difference
lat_pred<- read.csv("predict_GW/predicted_GW_aligned.csv")
lat_pred <- lat_pred %>% mutate(Diff= Prediction - Metadata) %>% arrange(desc(Diff))

# Summarize distributions
lat_pred  %>% filter(Diff> -1 & Diff < 1) %>%summary()

# Faceted view of both latitude types
lat_pred %>% pivot_longer(cols = c(Prediction, Metadata), names_to= "Type", values_to= "Latitude" ) %>%
  ggplot(aes(Latitude)) + geom_histogram(binwidth = 0.05, fill="skyblue") + facet_grid("Type") + ylab("Count") +
  ggtitle("Latitude Distributions for Two Label Types", subtitle= "Predicted for High Confidence Labels")

# Visualize difference
lat_pred %>% ggplot(aes(Diff)) + geom_histogram(binwidth=0.05, fill="red3") + ylab("Count") + 
  xlab("Difference between Prediction and Existing Label") + ggtitle("Distribution of Differences", subtitle="GW Labels") +
  geom_vline(xintercept=0.41)

#####
#
# INSPECTING OUTLIERS
#
#####

# Inspect samples with large difference
outliers<- lat_pred %>% filter(Diff> 1) %>% pull(Geno)

# Load PCA obj
pca<- readRDS("../../bio_findings/data/aug2021/pca/pca_noInland.RDS")

# Inspect PC %
pc.percent <- pca$varprop*100

# Create df for plotting pca
tab <- data.frame(Geno = pca$sample.id,
                  PC1 = pca$eigenvect[,1],   
                  PC2 = pca$eigenvect[,2],
                  PC3 = pca$eigenvect[,3],
                  PC4 = pca$eigenvect[,4],
                  PC5 = pca$eigenvect[,5],
                  stringsAsFactors = FALSE)
# Load meta data
meta<- readRDS("../../bio_findings/data/aug2021/5class_noInland.RDS")
samples<- meta %>%
  select(Geno, Latitude, Longitude, River, Class)

# Merge classes and meta
samples<- inner_join(meta, tab)

# Save dataframe
#write.table(samples, "../data/meta_pca_1254.csv", sep="\t", quote = F, row.names = F)

# Find outliers
outliers<- samples %>% filter(Geno %in% outliers)
williamette<- samples %>% filter(River=="Willamette")
nonGW_will<- williamette %>% filter(!grepl('GW', Geno))
GW_will<- williamette %>% filter(grepl('GW', Geno))
newNeigh<- samples %>% filter(Geno %in% new_neighbors)

# Plot PC
samples %>%
  ggplot(aes(PC1, PC2)) +
  geom_point() +
  ylab("PC2 (1.1% Var Explained)") +
  xlab("PC1 (2.5% Var Explained)") +
  ggtitle("WGS PCA", subtitle="High Confidence Label Outliers") +
  geom_point(data=nonGW_will, aes(PC1, PC2), color="red") +
  geom_point(data=GW_will, aes(PC1, PC2), color="skyblue") +
  geom_point(data=newNeigh, aes(PC1, PC2), color="lightgreen") 
  


### Plot faceted scatter for all GW river systems

# Isolate river systems of interest
gw_samples<- samples %>% filter(grepl("GW", Geno))
gw_rivers<- gw_samples %>% pull(River) %>% unique()

# Tag GW samples
rivers<- samples %>% filter(River %in% gw_rivers)
rivers$'GW Sample'<- grepl("GW", rivers$Geno)

# Faceted Plot
rivers %>% filter(River == "Willamette") %>%
  ggplot(aes(PC1, PC2, color= `GW Sample`)) +
  geom_point() +
  ylab("PC2 (1.1% Var Explained)") +
  xlab("PC1 (2.5% Var Explained)") +
  ggtitle("WGS PCA", subtitle="GW-tagged Samples")


#####
#
# HONE IN ON OUTLIERS
#
#####

# Tag outliers in full data
outmarked<- samples %>% mutate(Outlier= if_else(Geno %in% outliers, TRUE, FALSE))

# Visualize PCA
outmarked %>%
  ggplot(aes(PC1, PC2, color= `Outlier`)) +
  geom_point() +
  ylab("PC2 (1.1% Var Explained)") +
  xlab("PC1 (2.5% Var Explained)") +
  ggtitle("WGS PCA", subtitle="GW-tagged Samples")

# Zoom in on outlier PCA coordinates
xmin<- GW_will %>% select(PC1) %>% min()
xmax<- GW_will %>% select(PC1) %>% max()
ymin<-GW_will %>%  select(PC2) %>% min()
ymax<- GW_will %>%  select(PC2) %>% max()

# Build df of Willamette genotypes and genotypes clustering nearby in PCA
out_region<- samples %>% filter((PC1 > xmin & PC1< xmax & PC2>ymin & PC2<ymax) | River=="Willamette")
out_region<- out_region %>% mutate(River= if_else(River=="Willamette" & grepl('GW', Geno), 'GW_Willamette', River))

# Look at table summary of gps coordinates
out_region %>% group_by(River) %>% summarise(avg_lat= mean(Latitude), avg_long= mean(Longitude)) %>% arrange(avg_lat, avg_long)

# Do predicted latitude labels more accurately reflect PCA clustering?
gw_will_id<- out_region %>% filter(River=="GW_Willamette") %>% pull(Geno)
lat_pred %>% filter(Geno %in% gw_will_id) %>% summarise(avg_lat= mean(Prediction))

# Define new neighbors
new_neighbors<- out_region %>% filter(River %in% c("Skykomish", "Puyallup", "Skagit", "Nisqually", "Yakima")) %>% pull(Geno)


out_region %>%
  ggplot(aes(PC1, PC2, color=River)) +
  geom_point() +
  ylab("PC2 (1.1% Var Explained)") +
  xlab("PC1 (2.5% Var Explained)") +
  ylim(ymin, ymax) +
  xlim(xmin, xmax) +
  ggtitle("WGS PCA", subtitle="GW-tagged Samples")


















