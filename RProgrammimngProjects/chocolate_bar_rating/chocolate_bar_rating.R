# Chocolate Bar Rating Analysis
# Task
## * Where are the best cocoa beans grown?
## * Which countries produce the highest-rated bars?
## * What’s the relationship between cocoa solids percentage and rating?

# Loading packages

library(tidyverse)
library(skimr)    # for statistical summary
library(janitor)  # for examining and cleaning dirty data.
#install.packages("ggpubr")
# Install
#if(!require(devtools)) install.packages("devtools")
#devtools::install_github("kassambara/ggpubr")
#library("ggpubr")

# Import data

flavors_df <- read_csv("~/Desktop/huma/R/chocolate_bar_rating/flavors_of_cacao.csv")

# Exploring data

glimpse(flavors_df)
colnames(flavors_df)
head(flavors_df)

# Data Cleaning and organization

flavors_df %>%
  rename(brand = `Company(Maker-if known)`, bar_name = `Specific Bean Origin\ror Bar Name`, ref = REF, review_date = `Review\rDate`, 
         cocoa_percent = `Cocoa\rPercent`, company_location = `Company\rLocation`, 
         rating = Rating,  bean_type = `Bean\rType`, bean_origin = `Broad Bean\rOrigin`)

# or use clean_names()
flavors_df_v2 <- clean_names(flavors_df)
head(flavors_df_v2)

# select columns for analysis
trimmed_flavors_df <- flavors_df_v2 %>% 
  select(rating, cocoa_percent, bean_type, company_location) 
head(trimmed_flavors_df)

# find maximum rating
trimmed_flavors_df %>%
  summarize(max_rating = max(rating)) # it is 5

# consider any rating greater than or equal to 3.75 points can be a high rating. 
# considers a bar to be super dark chocolate if the bar's cocoa percent is greater than or equal to 80%.
best_trimmed_flavors_df <- trimmed_flavors_df %>% 
  filter(rating >= 3.75 & cocoa_percent >= 80)

as_tibble(best_trimmed_flavors_df)

# visualization

ggplot(data = best_trimmed_flavors_df) + 
  geom_bar(mapping = aes(x = company_location, color = rating), fill = "Lightgreen") +
  labs(title = "Countries Produce Highest-rated Bars", x = "Countries")

# Finding
# * According to bar chart, the two company locations that produce the highest rated chocolate bars are Canada and France. 

ggplot(data = best_trimmed_flavors_df) +
  geom_bar(mapping = aes(x = company_location, fill = rating)) + 
  facet_wrap(~ cocoa_percent) + 
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Countries Produce Highest rated Bars with Highest Cocoa Percent", x = "Company Location", fill = "Rating")

# Finding
# * Canada and France produce highest rated (4) bars with 88% and 80% cocoa solids.
# * USA produces chocolate bars with 3.75 rating and 80%, 82% and 90% cocoa solids.
# * Amsterdam and Scotland chocolate bars are with 80% cocoa and 3.75 rating.

#trimmed_flavors_df %>%
#  summarize(max(cocoa_percent), min(cocoa_percent))
# trimmed_flavors_df$cocoa_percent_num <- as.numeric(as.character(trimmed_flavors_df$cocoa_percent))
#is.numeric(cocoa_percent_num)

ggplot(data = trimmed_flavors_df) +
  geom_point(mapping = aes(x = rating, y = cocoa_percent, color = rating, alpha = 0.75)) + 
  labs(title = "Cocoa Percent vs Rating", x = "Rating", y = "Cocoa Percent", color = "Rating")

# Finding
# * There is no any direct correlation between cocoa percent and rating.

# converting cocoa percent into numeric
trimmed_flavors_df$cocoa_percent_num = as.numeric(gsub("[\\%,]", "", trimmed_flavors_df$cocoa_percent))

head(trimmed_flavors_df)
# Finding correlation between cocoa percent and rating
trimmed_flavors_df %>%
  summarize(cor = cor(rating,cocoa_percent_num)) # -0.165

# Visualization
ggplot(data = trimmed_flavors_df) +
  geom_point(mapping = aes(x = rating, y = cocoa_percent, color = rating, alpha = 0.75)) + 
  geom_smooth(aes(x = rating, y = cocoa_percent), method = lm, se = FALSE, fullrange = FALSE) +
  labs(title = "Cocoa Percent vs Rating", x = "Rating", y = "Cocoa Percent", color = "Rating")

# best cocoa origin
best_cocoa_origin <- flavors_df_v2 %>%
  select(rating,cocoa_percent,specific_bean_origin_or_bar_name, broad_bean_origin,bean_type, company_location) %>%
  filter(rating >= 3.75, cocoa_percent >= 80) %>%
  group_by(broad_bean_origin) %>%
  drop_na(bean_type)
head(best_cocoa_origin)
View(best_cocoa_origin)

ggplot(data = best_cocoa_origin) +
  geom_col(mapping = aes(x = broad_bean_origin, y = cocoa_percent, color = rating, fill = bean_type), 
           position = "dodge") +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Best Cocoa Bean Origin", x = "Bean Origin") 
# Finding
# * Ecuador produces best quality beans with rating 4 and 80% cocoa solids and bean types are Criollo and Trinitario.
# * Guatemala, Dominican Rep., Peru, Madagascar, Papua New Guinea also best producer with rating 4 and 88% cocoa.
# * Venezuela produces Criollo and Trinitario cocoa beans with 80% cocoa contents and rating 3.75.
# * Central and S.America are best in production of cocoa beans with 90% cocoa solids and rating 3.75
# * Peru produces Criollo bean type with 3.75 rating and 80% cocoa.
# * Costa Rica grows Matina beans with 82% cocoa and 3.75 rating.


# 
###################################
# Top Chocolate bar making companies
trimmed_flavors_df1 <- flavors_df_v2 %>% 
  select(rating, cocoa_percent, bean_type, company_location, company_maker_if_known) %>%
  rename(company_maker = company_maker_if_known)
head(trimmed_flavors_df1)

company_df <- trimmed_flavors_df1 %>%
  group_by(company_maker) %>%
  #count(company_maker, sort = TRUE)
  summarize(count = n()) %>%
  top_n(n = 5, wt = count)
View(company_df)
 # slice(1:5) %>%
  #summarize(company = count(company_maker)) %>%
  company_df %>%
    ggplot(aes(x = reorder(company_maker, -count), y = count, fill = company_maker)) +
    geom_col(position = "dodge") +
    theme(axis.text.x = element_text(angle = 90)) +
    geom_text(aes(label = count),vjust = -0.2, size = 3, angle = 0) +
    labs(title = "Chocolate Bar Making Companies", 
         x = "Company Name", fill = "Company")    
    
#ggplot(aes(x = fct_lump_n(fct_infreq(company_maker), n = 5))) +
#  geom_bar(position = "dodge", stat = "count") +
#  theme(axis.text.x = element_text(angle = 90)) +
#  labs(title = "Chocolate Bar Making Companies", subtitle = "Companies with rating", 
#       x = "Company Name")

# Bean type vs rating, and cocoa percent
trimmed_flavors_df2 <- trimmed_flavors_df1 %>%
  group_by(bean_type) %>%
  drop_na() %>%
  #mutate(bean_type = ifelse(is.na(bean_type), "Unknown", bean_type)) %>%
  summarize(count= n()) %>%
  top_n(n = 5, wt = count)
View(trimmed_flavors_df2)
ggplot(data = trimmed_flavors_df2, aes(x = reorder(bean_type, -count), y = count, fill = bean_type)) +
  geom_col(position = "dodge") +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Bean Types", 
       x = "Bean Type", y = "Count", fill = "Bean Type")


####################################################
best_trimmed_flavors_df1 <- trimmed_flavors_df1 %>% 
  filter(rating >= 3.75 & cocoa_percent >= 80)
as_tibble(best_trimmed_flavors_df1)

ggplot(data = best_trimmed_flavors_df1) +
  geom_bar(mapping = aes(x = company_maker, color = rating), fill = "purple")+
  labs(title = 'Choclate bar making company',subtitle= "Companies making best high content chocolate bar")+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 0.5))
# Pralus and Soma Companies making high content chocolate bar

# First task : Which countries produce the highest-rated bars?
best_origin_bean <- flavors_df_v2 %>% 
  filter(rating >=3.75, ref > 1700) %>% 
  arrange(-ref, -review_date) %>% 
  group_by(broad_bean_origin) %>% 
  drop_na(bean_type)
head(best_origin_bean)

#The first finding is about : Where are the best cocoa beans grown?
  
ggplot(best_origin_bean, mapping = aes(x = broad_bean_origin, fill = broad_bean_origin)) +
  geom_bar() + 
  theme(axis.text.x = element_text(angle = 90)) + 
  labs(title = "The Best Bean Origin")
  
#  First Finding :
    
#    The best cocoa beans are grown by : *
    
#    1. Ecuador *
    
#    2. Colombia *
    
#    3. Nicaragua *
  
arranged_data <-  flavors_df_v2 %>% 
  filter(rating >= 3.75) %>% 
  group_by(company_location) %>% 
  drop_na()
head(arranged_data)
  
arranged_data <-  flavors_df_v2 %>% 
  select(company_location, cocoa_percent, rating, review_date) %>% 
  filter(rating >= 3.75, cocoa_percent >=70) %>% 
  arrange(-rating, -review_date) %>%  
  group_by(company_location)
head(arranged_data)

ggplot(data = arranged_data, mapping = aes( x = company_location)) + 
  geom_bar() + 
  theme(axis.text.x = element_text(angle = 90)) + 
  labs(title = "Higher rated countries producing cocoa")
  
#  Second FINDING :
    
#    Countries producing higher rated bars are
  
#  1. U.S.A.
  
#  2. France
  
#  3. canada
  
#  4. Italy just(2006 & 2007)
  
# Final task is to know the relationship between cocoa solids percentage and rating
  
cor_flavor_of_cacao <-  flavors_df_v2 %>%  
  select(rating, cocoa_percent) %>% 
  arrange(+rating, cocoa_percent)
head(cor_flavor_of_cacao)

ggplot(cor_flavor_of_cacao, mapping = aes( x = rating, y = cocoa_percent)) + geom_point()

# This vizualization shows that there is no association between the two variables (Rating and Cocoa percentage)
  
# rmarkdown::render("chocolate_bar_rating.R", "pdf_document")
