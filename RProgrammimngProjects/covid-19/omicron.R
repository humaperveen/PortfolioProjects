# Covid-19 Variants Data Analysis
## Omicron variant Analysis
# Loading packages
library(tidyverse)
library(lubridate)
library(skimr)    # for statistical summary
library(janitor) # for examining and cleaning dirty data.
library(scales)
install.packages("maps", repos = "http://cran.us.r-project.org")
library(maps)

# Import data
covid_df <- read_csv("~/Desktop/huma/R/covid/covid-variants.csv")

# Exploring data
str(covid_df)

head(covid_df)

skim_without_charts(covid_df)

# Rename column
covid_df <- covid_df %>%
  select(location, date, variant, num_sequences, perc_sequences, num_sequences_total) %>%
  rename(num_seq_processed = num_sequences, perc_seq_processed = perc_sequences)

View(covid_df)
colnames(covid_df)

# Count unique variants
covid_df %>% tabyl(variant) %>%
  adorn_pct_formatting(digits = 0, affix_sign = TRUE)

# summarize variants
covid_df %>% summarize(variant, num_seq_processed, perc_seq_processed,num_sequences_total)

covid_df %>% 
  group_by(variant) %>%
  summarize( total_seq_processed = sum(num_seq_processed), total_perc_processed = sum(perc_seq_processed), total_seq = sum(num_sequences_total)) %>%
  head(24)

# Unique Variants
Uniq_variant = as.list(unique(covid_df$variant))
Uniq_variant

# group by variant
variant_df <- covid_df %>%
  group_by(variant) %>%
  summarize(total_seq_processed = sum(num_seq_processed))

head(variant_df)

# variant plot 
# options(scipen=999)  # turn off scientific notation like 1e+06
 options(repr.plot.width = 18, repr.plot.height = 6)
# variant_df %>%
#  mutate(place = if_else(row_number() == 13, 1, 0)) %>%
ggplot(data = variant_df, mapping = aes(x = total_seq_processed, y = variant, fill = variant)) + 
  geom_col(position = "dodge", color = "Black") + 
  geom_text(aes(label = total_seq_processed),hjust = -0.05, nudge_x = 0.5, size = 2.5, angle = 0) +
  labs(title = "COVID Variants vs Total number of Sequences Processed", x = "Total Sequences Processed", y = "Covid Variants") +
  scale_x_continuous(labels = scales::comma, limit = c(0, 4500000)) + 
  theme_minimal()
#theme(axis.text.x = element_text(angle=90)) +

#ggplot(data = variant_df) + 
#  geom_col(mapping = aes(x = variant, y = total_seq_processed, fill = variant), position = "dodge", color = "Black") + 
#  geom_text(aes(x = variant, y = total_seq_processed, label = total_seq_processed),vjust = -0.5, label.size = 0.25) +
#  labs(title = "COVID Variants vs Total number of Sequences Processed", x = "Total Sequences Processed", y = "Covid Variants") +
#  scale_y_continuous(labels = scales::comma) +
#  theme(axis.text.x = element_text(angle=90))
options(repr.plot.width = 18, repr.plot.height = 8)
variant_df %>%
  filter(total_seq_processed < 100000) %>%
ggplot(aes(x = variant, y = total_seq_processed, fill = variant)) + 
  geom_col(position = "dodge", color = "Black") + 
  geom_text(aes(label = total_seq_processed),vjust = -0.2, nudge_y = 0.5, size = 4, angle = 0) +
  labs(title = "COVID Variants vs Total number of Sequences Processed < 100,000", x = "Covid Variants", y = "Total Sequences Processed", fill = "Covid Variants") +
  scale_y_continuous(labels = scales::comma, limits = c(0, NA)) + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle=90)) + 
  theme(axis.text = element_text(size = 8)) +
  theme(axis.title = element_text(size = 10), title = element_text(size = 10))

variant_df %>%
  filter(total_seq_processed > 100000) %>%
  ggplot(aes(x = variant, y = total_seq_processed, fill = variant)) + 
  geom_col(position = "dodge", color = "Black") + 
  geom_text(aes(label = total_seq_processed), vjust = -0.2, nudge_y = 0.5, size = 4, angle = 0) +
  labs(title = "COVID Variants vs Total number of Sequences Processed > 100,000", x = "Covid Variants", y = "Total Sequences Processed", fill = "Covid Variants") +
  scale_y_continuous(labels = scales::comma, limits = c(NA, NA)) + 
  theme_minimal() +
  theme(axis.text = element_text(size = 8)) +
  theme(axis.title = element_text(size = 10), title = element_text(size = 10))

# group by variant and location
loc_variant_df <- covid_df %>%
  group_by(location, variant) %>%
  summarize(total_seq_processed = sum(num_seq_processed))
View(loc_variant_df)

loc_variant_df1 <- covid_df %>%
  group_by(location, variant) %>%
  summarize(total_seq_processed = sum(num_seq_processed)) %>%
  filter(total_seq_processed >0)
View(loc_variant_df1)

loc_variant_df2 <- covid_df %>%
  group_by(location) %>%
  summarize(total_seq_processed = sum(num_seq_processed)) %>%
  filter(total_seq_processed >0) %>%
  arrange(desc(total_seq_processed))
View(loc_variant_df2)

#loc_variant_df2<- aggregate(loc_variant_df1$total_seq_processed ~ loc_variant_df1$location, FUN = sum)
#View(loc_variant_df2)

# variant, location plot
# variant, location plot for top 20 countries
options(repr.plot.width = 15, repr.plot.height = 30)
loc_variant_df2 %>%
  head(20) %>%
  group_by(location) %>%
  ggplot(mapping = aes(x = total_seq_processed, y = reorder(location, total_seq_processed), fill = total_seq_processed)) + 
  geom_col(position = "dodge", color = "Black") + 
  scale_fill_gradient(low = "#132B43", high = "#56B1F7", space = "Lab", na.value = "gray50", guide = "colourbar", aesthetics = "fill") +
  geom_text(aes(label = total_seq_processed),hjust = -0.1, nudge_x = 0.5, size = 4, angle = 0) +
  labs(title = "COVID Variants vs Total number of Sequences Processed for Top 20 Countries", x = "Total Sequences Processed", y = "Countries", fill = "Total Seq. Processed") +
  scale_x_continuous(labels = scales::comma, limit = c(0, 3000000)) + 
  theme(axis.text = element_text(size = 8)) +
  theme(axis.title = element_text(size = 10), title = element_text(size = 10))
# or change scale fill gradient
options(repr.plot.width = 15, repr.plot.height = 30)
loc_variant_df2 %>%
  head(20) %>%
  group_by(location) %>%
  ggplot(mapping = aes(x = total_seq_processed, y = location, fill = total_seq_processed)) + 
  geom_col(position = "dodge", color = "Black") + 
  scale_fill_gradient(low = "Yellow", high = "Red", space = "Lab", na.value = "gray50", guide = "colourbar", aesthetics = "fill") +
  geom_text(aes(label = total_seq_processed),hjust = -0.1, nudge_x = 0.5, size = 4, angle = 0) +
  labs(title = "COVID Variants vs Total number of Sequences Processed for Top 20 Countries", x = "Total Sequences Processed", y = "Countries") +
  scale_x_continuous(labels = scales::comma, limit = c(0, 3000000)) +
  theme(axis.text = element_text(size = 8)) +
  theme(axis.title = element_text(size = 10), title = element_text(size = 10))
#theme(axis.text.x = element_text(angle=90)) +
#filter(!is.na(loc_variant_df$total_seq_processed)) %>%
#  filter(total_seq_processed > 0) %>%

# variant, location plot
options(repr.plot.width = 15, repr.plot.height = 40)
loc_variant_df1 %>%
  group_by(location, variant) %>%
  ggplot(mapping = aes(x = total_seq_processed, y = location, fill = variant)) + 
  geom_bar(stat = "identity") + 
  # geom_text(aes(label = total_seq_processed),hjust = -0.1, nudge_x = 0.5, size = 2.5, angle = 0) +
  labs(title = "Countrywise COVID Variants vs Total number of Sequences Processed", x = "Total Sequences Processed", y = "Location", fill = "Covid Variant") +
  scale_x_continuous(labels = scales::comma) + 
  theme(axis.text.x = element_text(size = 4)) +
  theme(axis.text = element_text(size = 8)) +
  theme(axis.title = element_text(size = 10), title = element_text(size = 10))
  #theme(legend.position = "none")

# filter Omicron variant
omicron_variant_df <- covid_df %>%
  filter(variant == "Omicron")

View(omicron_variant_df)

# Omicron data group by location
omicron_variant_df <- omicron_variant_df %>%
  group_by(location) %>%
  summarize(total_seq_processed = sum(num_seq_processed)) %>%
  arrange(desc(total_seq_processed))

head(omicron_variant_df)

# plot
options(repr.plot.width = 15, repr.plot.height = 30)
covid_df %>%
  filter(variant == "Omicron") %>%
  group_by(location) %>%
  summarize(total_seq_processed = sum(num_seq_processed)) %>%
  ggplot(aes(x = total_seq_processed, y = location)) + 
  geom_col(position = "dodge",fill = "Purple" , color = "Black") + 
  labs(title = "Countrywise Omicron COVID Variant vs Total number of Sequences Processed", x = "Total Sequences Processed", y = "Countries", fill = "Countries") +
  scale_x_continuous(labels = scales::comma) +
  theme(axis.text = element_text(size = 8)) +
  theme(axis.title = element_text(size = 10), title = element_text(size = 10))

covid_df %>%
  filter(variant == "Omicron") %>%
  group_by(location) %>%
  summarize(total_seq_processed = sum(num_seq_processed)) %>%
  arrange(desc(total_seq_processed)) %>%
  head(10) %>%
  ggplot(aes(x = total_seq_processed, y = location)) + 
  geom_col(position = "dodge",fill = "Purple" , color = "Black") + 
  geom_text(aes(label = total_seq_processed),hjust = -0.1, nudge_x = 0.5, size = 2.5, angle = 0) +
  labs(title = "Omicron COVID Variant vs Total number of Sequences Processed in Top 10 Countries", x = "Total Sequences Processed", y = "Countries", fill = "Countries") +
  scale_x_continuous(labels = scales::comma, limit = c(0, 80000)) + 
  theme(axis.text = element_text(size = 8)) +
  theme(axis.title = element_text(size = 10), title = element_text(size = 10))

# Omicron data location vs total number of sequences
#omicron_variant_df <- omicron_variant_df %>%
#  group_by(location) %>%
#  summarize(total_seq = sum(num_sequences_total))
  #arrange(desc(total_seq))

#head(omicron_variant_df)

# Omicron total number of sequences by countries
omicron_variant_df <- covid_df %>%
  filter(variant == "Omicron")

omicron_variant_df_v2 <-aggregate(omicron_variant_df$num_sequences_total ~ omicron_variant_df$location, FUN = sum)

View(omicron_variant_df_v2)

# Omicron total number of sequences by countries
omicron_variant_df <- covid_df %>%
  filter(variant == "Omicron") %>%
  group_by(location) %>%
  summarize(total_seq = sum(num_sequences_total))
#arrange(desc(total_seq))

head(omicron_variant_df)

# Omicron total number of sequences by countries in desc order
total_omicron_variant_df <- covid_df %>%
  filter(variant == "Omicron") %>%
  group_by(location) %>%
  summarize(total_seq = sum(num_sequences_total)) %>%
  arrange(desc(total_seq))

head(total_omicron_variant_df)
View(total_omicron_variant_df)

# Plotting the map data
world_data = map_data("world") 
head(world_data)

# renaming location to region
omicron_df <- total_omicron_variant_df %>% rename(region = location)
head(omicron_df)

# Plotting the map
world_data = left_join(world_data, omicron_df, by="region")
world_data1 = world_data %>% filter(!is.na(world_data$total_seq))
world_data2 = world_data1 %>%  filter(total_seq > 0)
# plot
options(repr.plot.width = 22, repr.plot.height = 12)
ggplot(world_data2, aes(x = long, y = lat, group=group)) +
  geom_polygon(aes(fill = total_seq), color = "black")+
  #scale_fill_gradient2(name = "total_seq", low = "yellow", mid = "white", 
#                       high = "red", midpoint = 0, space = "Lab", na.value = "gray50", 
#                       guide = "colourbar", aesthetics = "fill")+
  scale_fill_viridis_c(option = "plasma", trans = "sqrt") +
  theme_minimal()+
  theme(axis.text = element_text(size = 18)) +
  theme(axis.title = element_text(size = 20), title = element_text(size = 25)) + 
  labs(title = "Omicron Variant Total Sequences", fill = "Total Sequences") 
  
# plot
  options(repr.plot.width = 22, repr.plot.height = 12)
  ggplot(world_data1, aes(x = long, y = lat, group=group)) +
    geom_polygon(aes(fill = total_seq), color = "black")+
    scale_fill_gradient2(name = "total_seq", low = "white",high = "red")+
    theme_minimal()+
    theme(axis.text = element_text(size = 15)) +
    theme(axis.title = element_text(size = 17))