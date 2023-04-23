## Case Study 1
## Case Study: How Does a Bike-Share Navigate Speedy Success?
## Scenario
# I am a junior data analyst working in the marketing analyst team at Cyclistic, a bike-share company in Chicago. 
# The director of marketing believes the company’s future success depends on maximizing the number of annual memberships. 
# Therefore, my team wants to understand how casual riders and annual members use Cyclistic bikes differently. 
# From these insights, my team will design a new marketing strategy to convert casual riders into annual members.

## About the company
# In 2016, Cyclistic launched a successful bike-share offering. Since then, the program has grown to a fleet of 5,824 bicycles 
# that are geotracked and locked into a network of 692 stations across Chicago. The bikes can be unlocked from one station and returned to any other station in the system anytime.

## Ask
# Guiding questions: How do annual members and casual riders use Cyclistic bikes differently?
# Business tasks: Design marketing strategies aimed at converting casual riders into annual members.
# Key stakeholders: here stakeholder is Lily Moreno: The director of marketing and your manager. Moreno is responsible for the development of 
# campaigns and initiatives to promote the bike-share program. These may include email, social media, and other channels.
# I will produce a report with the following deliverables:
# 1. A clear statement of the business task
# 2. A description of all data sources used
# 3. Documentation of any cleaning or manipulation of data
# 4. A summary of your analysis
# 5. Supporting visualizations and key findings
# 6. Your top three recommendations based on your analysis

## Prepare
# I will use Cyclistic’s historical trip data from Dec2020- Nov2021 to analyze and identify trends.
# For the purposes of this case study,[Download the previous 12 months of Cyclistic trip data here.] the datasets are appropriate and will enable me to answer the business questions. 
# The data has been made available by Motivate International Inc. under this [license.](https://ride.divvybikes.com/data-license-agreement) 
# This is public data that you can use to explore how different customer types are using Cyclistic bikes. 
# Data is example of ROCCC because it is original data from a reliable organization, comprehensive, current, and cited.
# I downloaded and stored data appropriately. I follow the file naming convention appropriately too.

# Step 1: Install and Load required packages
install.packages("tidyverse") # for data manipulation, exploration, and visualization.
library(tidyverse)
install.packages("lubridate") # for date functions
library(lubridate)
install.packages("scales")
library(scales)
getwd() #displays your working directory
setwd("Desktop/Huma/Course8/case_study1/source") # #sets working directory to simplify calls to data

# Step 2: Import Data
# 12 months trip data from Dec,2020 to Nov,2021
tripdata_2020_12 <- read_csv("202012-divvy-tripdata.csv")
tripdata_2021_01 <- read_csv("202101-divvy-tripdata.csv")
tripdata_2021_02 <- read_csv("202102-divvy-tripdata.csv")
tripdata_2021_03 <- read_csv("202103-divvy-tripdata.csv")
tripdata_2021_04 <- read_csv("202104-divvy-tripdata.csv")
tripdata_2021_05 <- read_csv("202105-divvy-tripdata.csv")
tripdata_2021_06 <- read_csv("202106-divvy-tripdata.csv")
tripdata_2021_07 <- read_csv("202107-divvy-tripdata.csv")
tripdata_2021_08 <- read_csv("202108-divvy-tripdata.csv")
tripdata_2021_09 <- read_csv("202109-divvy-tripdata.csv")
tripdata_2021_10 <- read_csv("202110-divvy-tripdata.csv")
tripdata_2021_11 <- read_csv("202111-divvy-tripdata.csv")

# check column name should be consistent

colnames(tripdata_2020_12)
colnames(tripdata_2021_01)
colnames(tripdata_2021_02)
colnames(tripdata_2021_03)
colnames(tripdata_2021_04)
colnames(tripdata_2021_05)
colnames(tripdata_2021_06)
colnames(tripdata_2021_07)
colnames(tripdata_2021_08)
colnames(tripdata_2021_09)
colnames(tripdata_2021_10)
colnames(tripdata_2021_11)

# check the structure of dataframes for any incongruencies
str(tripdata_2020_12)
str(tripdata_2021_01)
str(tripdata_2021_02)
str(tripdata_2021_03)
str(tripdata_2021_04)
str(tripdata_2021_05)
str(tripdata_2021_06)
str(tripdata_2021_07)
str(tripdata_2021_08)
str(tripdata_2021_09)
str(tripdata_2021_10)
str(tripdata_2021_11)

# Combine all dataframes into one dataframe for one year

all_tripdata <- bind_rows(tripdata_2020_12, tripdata_2021_01, tripdata_2021_02, tripdata_2021_03, 
                          tripdata_2021_04, tripdata_2021_05, tripdata_2021_06, tripdata_2021_07, 
                          tripdata_2021_08, tripdata_2021_09, tripdata_2021_10, tripdata_2021_11)

# write.csv(all_tripdata, file = "~/Desktop/Huma/Course8/case_study1/my_work/bike_ridership_data_dec20_nov21.csv")
# Remove start_lat, start_lng, end_lat and end_lng columns because I dont need these feature in my data analysis.

all_trips <- all_tripdata %>%
  select(-c(start_lat, start_lng, end_lat, end_lng))

## Process
# data cleaning and add data for analysis 
# Due to the large dataset I will use R for data cleaning and analysis

# list all column names

colnames(all_trips) 

# dimensions of data frame

dim(all_trips)    # 5479096       9

# number of rows in data frame

nrow(all_trips)   # 5479096

# Only show first 6 rows of data frame

head(all_trips)

# only last 6 rows of data frames

tail(all_trips)

# Brief structure of data frame with column name and data type

str(all_trips)

# summary of data frame, mainly numerics

summary(all_trips)

# now I need to add column for date, month, day, year, and day of week for each ride  to aggregate
# ride data for each day, month and year

all_trips$date <- as.Date(all_trips$started_at) # default date format is yyyy-mm-dd
all_trips$month <- format(as.Date(all_trips$date), "%B")
all_trips$day <- format(as.Date(all_trips$date), "%d")
all_trips$year <- format(as.Date(all_trips$date), "%Y")
all_trips$day_of_week <- format(as.Date(all_trips$date), "%A")
all_trips$hour <- format(as_datetime(all_trips$started_at), "%H")

# confirm that column has been added

colnames(all_trips)

# now I add ride_length to all_trips data frame to calculate ride length in seconds for each trip

all_trips$ride_length <- difftime(all_trips$ended_at, all_trips$started_at)


# inspect the data frame again to check column is added or not and what is data type of ride_length

str(all_trips)

# Convert "ride_length" to numeric so we can run calculations on the data

all_trips$ride_length <- as.numeric(as.character(all_trips$ride_length))
is.numeric(all_trips$ride_length)

all_trips$hour <- as.numeric(as.character(all_trips$hour))
is.numeric(all_trips$hour)
str(all_trips)

# Remove bad data 
# remove data where ride length is 0 or less than 0 and when bikes were taken out of docks and checked for quality
# more than 1440 minutes because ride length should not be either negative or more than one day

all_trips_v2 <- all_trips[!(all_trips$rideable_type =="docked_bike" | all_trips$ride_length <= 0),]
nrow(all_trips_v2) # 5157877 

# all_trips_v2 <- all_trips[!(all_trips$ride_length <= 0 | all_trips$ride_length > 1440),]
# nrow(all_trips_v2) #4272114 

# all_trips_v2 <- all_trips[!(all_trips$start_station_name == "HQ QR" | all_trips$ride_length <= 0),]
# nrow(all_trips_v2) # 5478022

# check new data frame

dim(all_trips_v2)
View(all_trips_v2)
summary(all_trips_v2)

# Remove NA from new data frame

all_trips_v3 <- drop_na(all_trips_v2)

dim(all_trips_v3)  # 3508743      15    # 4205404  16
View(all_trips_v3)
summary(all_trips_v3)

write.csv(all_trips_v3, file = "~/Desktop/Huma/Course8/case_study1/my_work/cleaned_bike_ridership_data_dec20_nov21.csv")
# check to remove duplicates from data frame

# all_trips_v4 <- all_trips_v3[!duplicated(all_trips_v3$ride_id),]

# dim(all_trips_v4)   # 3508743      15
# View(all_trips_v4)

#Find out the distance for each ride:
#all_trips_v3$ride_distance <- distGeo(matrix(c(all_trips_v3$start_lng, all_trips_v3$start_lat), ncol = 2), matrix(c(all_trips_v3$end_lng, all_trips_v3$end_lat), ncol = 2))
#View(all_trips_v3)

## Analyze
# Descriptive analysis on ride length

# avg_ride_length <- mean(all_trips_v3$ride_length)
# median_ride_length <- median(all_trips_v3$ride_length)
# max_ride_length <- max(all_trips_v3$ride_length)
# min_ride_length <- min(all_trips_v3$ride_length)

# Descriptive analysis on ride_length (all figures in seconds)
mean(all_trips_v3$ride_length) #straight average (total ride length / rides)
median(all_trips_v3$ride_length) #midpoint number in the ascending array of ride lengths
max(all_trips_v3$ride_length) #longest ride
min(all_trips_v3$ride_length) #shortest ride

# You can condense the four lines above to one line using summary() on the specific attribute
summary(all_trips_v3$ride_length)

# Compare ride length by members and casual users
aggregate(all_trips_v3$ride_length ~ all_trips_v3$member_casual, FUN = mean)
aggregate(all_trips_v3$ride_length ~ all_trips_v3$member_casual, FUN = median)
aggregate(all_trips_v3$ride_length ~ all_trips_v3$member_casual, FUN = max)
aggregate(all_trips_v3$ride_length ~ all_trips_v3$member_casual, FUN = min)

#Assign the correct order to each day of the week
all_trips_v3$day_of_week <- 
  ordered(all_trips_v3$day_of_week, levels = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))

# See the average ride time by each day for members vs casual users
aggregate(all_trips_v3$ride_length ~ all_trips_v3$member_casual + all_trips_v3$day_of_week, FUN = mean)

#all_trips_v3 %>% 
#  group_by(member_casual, day_of_week) %>%
#  summarize(avg_ride_length = mean(ride_length), .group = "drop") %>%
#  arrange(day_of_week)

# now find number of rides per day for members and casual users

all_trips_v3 %>%
  group_by(member_casual, day_of_week) %>%
  summarize(num_of_rides = n(), .groups = "drop") %>%
  arrange(day_of_week)

# analyze ridership data by riders type and weekday
all_trips_v3wd <- all_trips_v3 %>%
  mutate(weekday = wday(started_at, label = TRUE)) %>%  #creates weekday field using wday()
  group_by(member_casual, weekday) %>%  #groups by usertype and weekday
  summarise(number_of_rides = n()							#calculates the number of rides and average duration
            ,average_duration = mean(ride_length)) %>% 		# calculates the average duration
  arrange(member_casual, weekday)								# sorts

View(all_trips_v3wd)
write.csv(all_trips_v3wd, file = "~/Desktop/Huma/Course8/case_study1/my_work/weeky_ridership_for_member_casuals.csv")
#Assign the correct order to each month of the year

# all_trips_v3$month <-
#  ordered(all_trips_v3$month, levels = c('12', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'))

all_trips_v3$month <-
  ordered(all_trips_v3$month, levels = c("December", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November"))
# then convert to factor
#all_trips_v3$month = factor(all_trips_v3$month, levels = month.abb)

# find monthly rides by members and casual users

all_trips_v3 %>%
  group_by(member_casual, month) %>%
  summarise(number_of_ride = n(), .groups = 'drop') %>%
  arrange(month)

# analyze ridership data by riders type and month
all_trips_v3m <- all_trips_v3 %>%
  group_by(member_casual, month) %>%  #groups by usertype and months
  summarise(number_of_rides = n()							#calculates the number of rides and average duration
            ,average_duration = mean(ride_length)) %>% 		# calculates the average duration
  arrange(member_casual, month)								# sorts

View(all_trips_v3m)
write.csv(all_trips_v3m, file = "~/Desktop/Huma/Course8/case_study1/my_work/monthly_ridership_for_member_casuals.csv")
# Findings:

# *Casual riders are more likely to take a ride on weekend while membership riders use on weekday more often.
# *Summer is the peak season for both rider types


## now lets find the average ride time by each month for members vs casual users
aggregate(all_trips_v3$ride_length ~ all_trips_v3$member_casual + all_trips_v3$month, FUN = mean)
# or
# all_trips_v3 %>%
#  group_by(member_casual, month) %>%
#  summarise(average_ride_length = mean(ride_length), .groups = 'drop') %>%
#  arrange(month)

#Findings:
  
# *Membership rider's trip is longer than casual ones regardless of the season or day
# *All users take longer trips over weekend and summer

# number of rides by members or casual users

all_trips_v3 %>%
  group_by(member_casual) %>%
  summarize(number_of_rides = n() , .groups = 'drop')

# hourly analysis for ridership for casual riders
all_trips_v4 <- all_trips_v3 %>%
  group_by(member_casual, hour) %>%
  summarize(number_of_rides = n(), average_duration = mean(ride_length), .groups = 'drop') %>%
  filter(member_casual =="casual")
  
  #arrange(desc(number_of_rides))

max(all_trips_v4$number_of_rides)
min(all_trips_v4$number_of_rides)
mean(all_trips_v4$number_of_rides)
median(all_trips_v4$number_of_rides)
View(all_trips_v4) 
# head(all_trips_v4)
# tail(all_trips_v4)

write.csv(all_trips_v4, file = "~/Desktop/Huma/Course8/case_study1/my_work/hourly_ridership_for_casuals.csv")

# hourly analysis for riders type for member riders
all_trips_v5 <- all_trips_v3 %>%
   group_by(member_casual, hour) %>%
   summarize(number_of_rides = n(),average_duration = mean(ride_length), .groups = 'drop') %>%
   filter(member_casual =="member")
 
#arrange(desc(number_of_rides))
 
 max(all_trips_v5$number_of_rides)
 min(all_trips_v5$number_of_rides)
 mean(all_trips_v5$number_of_rides)
 median(all_trips_v5$number_of_rides)
 View(all_trips_v5)

#write_csv(all_trips_v5, "Desktop/Huma/Course8/case_study1/my_work/hourly_ridership_for_members.csv") 
write.csv(all_trips_v5, file = "~/Desktop/Huma/Course8/case_study1/my_work/hourly_ridership_for_members.csv")

## Visualizations

# visualization of numbers of rides by members and casual users

all_trips_v3 %>%
  group_by(member_casual) %>%
  summarize(number_of_rides = n() , .groups = 'drop') %>%
  ggplot(aes(x = member_casual, y = number_of_rides, fill = member_casual)) +
  geom_col(position = "dodge") + 
  labs(title = "Number of Rides by Rider Types", x = "Riders Type", y = "Number of Rides", fill = "Riders Type") +
  scale_y_continuous(labels = scales::comma)
  
# visualization of number of rides by days of week for riders type
all_trips_v3 %>%
  mutate(weekday = wday(started_at, label = TRUE)) %>%
  group_by(member_casual, weekday) %>%
  summarize(number_of_rides = n() , .groups = 'drop') %>%
  ggplot(aes(x = weekday, y = number_of_rides, fill = weekday)) +
  geom_col(position = "dodge") + 
  facet_wrap(~ member_casual) +
  labs(title = "Number of Rides Grouped by Day of Week", subtitle = "Faceted by Rider Types", 
       x = "Days of Week", y = "Number of Rides", fill = "Days of Week") + 
  scale_y_continuous(labels = scales::comma)
# theme(axis.text.x = element_text(angle = 65)) +
## Let's visualize the number of rides per day by rider type

# all_trips_v3 %>%
#  mutate(weekday = wday(started_at, label = TRUE)) %>%
#  group_by(member_casual, weekday) %>%
#  summarise(number_of_rides = n()) %>%
#  arrange(member_casual, weekday) %>%
#  ggplot(aes(x = weekday, y = number_of_rides, fill = member_casual)) +
#  geom_col(position = "dodge") +
#  theme(axis.text.x = element_text(angle = 65)) +
#  labs(title = "Number of Rides vs Day of Week")

# visualization of monthly number of rides by riders type
all_trips_v3 %>%
  group_by(member_casual, month) %>%
  summarize(number_of_rides = n(), .groups = 'drop') %>%
  arrange(month) %>%
  ggplot(aes(x = month, y = number_of_rides, fill = member_casual)) + 
  geom_bar(position = "dodge", stat = "identity") + 
  theme(axis.text.x = element_text(angle = 65)) +
  labs(title = "Monthly Number of Rides by Members and Casuals", 
       x = "Months", y = "Number of Rides", fill = "Riders Type") +
  scale_y_continuous(labels = scales::comma)

# create visualization for number of rides by hours of the day for riders type

all_trips_v3 %>%
  group_by(member_casual, hour) %>%
  summarize(number_of_rides = n(), .group = "drop") %>%
  ggplot(aes(x = hour, y = number_of_rides, fill = member_casual)) +
  geom_col(position = "dodge") +
  facet_wrap(~ member_casual) +
  scale_x_time(breaks = c(0, 3, 6, 9, 12, 15, 18, 21), 
               labels = c("12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM")) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Numbers of Rides by Hours of Day", subtitle = "Faceted by Riders Type",
       x = "Hours", y = "Number of Rides", fill = "Riders Type") +
  theme(legend.position = "none")

all_trips_v3 %>%
  group_by(member_casual, hour) %>%
  summarize(number_of_rides = n(), .group = "drop") %>%
  ggplot(aes(x = hour, y = number_of_rides, color = member_casual)) +
  geom_line() +
  scale_x_time(breaks = c(0, 3, 6, 9, 12, 15, 18, 21), 
               labels = c("12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM")) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Numbers of Rides by Hours of Day", 
       x = "Hours", y = "Number of Rides", color = "Riders Type") 
  
#scale_x_continuous(breaks = c(0, 3, 6, 9, 12, 15, 18, 21) 
#, labels = c("12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM")) +
##geom_line() +
  #Here is where I format the x-axis
##  scale_x_datetime(labels = date_format("%Y-%m-%d %H"),
##                   date_breaks = "8 hours")

  
##mutate(month = factor(month.abb[all_trips_v3$month], levels = month.abb))

# Let's create a visualization for average ride length per day by riders type

all_trips_v3 %>%
  mutate(weekday = wday(started_at, label = TRUE)) %>%
  group_by(member_casual, weekday) %>%
  summarize(average_ride_length = mean(ride_length)) %>%
  arrange(member_casual, weekday) %>%
  ggplot(aes(x = weekday, y = average_ride_length, fill = weekday)) +
  geom_col(position = "dodge") +
  facet_wrap(~ member_casual) +
  labs(title = "Average Ride Length by Days of Week", subtitle = "Faceted by Riders Type",
       x = "Days of Week", y = "Average Ride Length(Seconds)", fill = "Days of Week") +
  scale_y_continuous(labels = scales::comma)

# Let's create a visualization for average ride length per month by riders type

all_trips_v3 %>%
  group_by(member_casual, month) %>%
  summarize(average_ride_length = mean(ride_length)) %>%
  arrange(member_casual, month) %>%
  ggplot(aes(x = month, y = average_ride_length, fill = member_casual)) +
  geom_col(position = "dodge") +
  theme(axis.text.x = element_text(angle = 45)) +
  labs(title = "Average Ride Length From Dec,2020 To Nov,2021", x = "Months", 
       y = "Average Ride Length(Seconds)", fill = "Riders Type") +
  scale_y_continuous(labels = scales::comma)

# Visualization for average ride length per hour by riders type
all_trips_v3 %>%
  group_by(member_casual, hour) %>%
  summarize(average_ride_length = mean(ride_length)) %>%
  ggplot(aes(x = hour, y = average_ride_length, fill = member_casual)) +
  geom_col(position = "dodge") +
  facet_wrap(~ member_casual) +
  scale_x_time(breaks = c(0, 3, 6, 9, 12, 15, 18, 21), 
               labels = c("12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM")) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Average Ride Length by Hours of Day", subtitle = "Faceted by Riders Type",
       x = "Hours", y = "Average Ride Length(Seconds)", fill = "Riders Type") +
  theme(legend.position = "none")

all_trips_v3 %>%
  group_by(member_casual, hour) %>%
  summarize(average_ride_length = mean(ride_length)) %>%
  ggplot(aes(x = hour, y = average_ride_length, color = member_casual)) +
  geom_line() +
  scale_x_time(breaks = c(0, 3, 6, 9, 12, 15, 18, 21), 
               labels = c("12 AM", "3 AM", "6 AM", "9 AM", "12 PM", "3 PM", "6 PM", "9 PM")) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "Average Ride Length by Hours of Day",
       x = "Hours", y = "Average Ride Length(Seconds)", color = "Riders Type")

## Analysis:

# from my analysis, it's clear that casual riders have less number of rides than members but their average ride length (duration in seconds) is comparatively more than members.
# from hourly data, we can say that casuals ride length has sharp rise around 7:30 AM till 10:00 AM while membership riders have almost constant average ride length.
# Another interesting finding is that casuals are more active at week end in compare to members who are using more bike rides on weekdays.

## Conclusion:

# To attract casual users, short term membership over weekends can be good option.
# membership discounts can also be planned for summer and autumn months (from May to October).


#It seems that the casual users travel the same average distance than the member users, but they have relatively longer rides, that would indicate a more leisure oriented usage vs a more "public transport" or pragmatic use of the bikes by the annual members.
#Casual riders are more likely to return their bikes at the same station.
#Additionaly, while that membership riders are more active on weekday, casual riders use the service more often over weekend. It lead me to conclude that membership riders use this service for their commute while casual rider use it for fun.

## Conclusion:
  
# 1)The Casual users have leisure, and tourism rides mostly on weekends.

# 2)The Annual users have commute or pragmatic rides during weekdays.

## References
# 1. Google. (n.d.). Google Data Analytics Professional Certificate [MOOC]. Coursera. https://www.coursera.org/professional-certificates/google-data-analytics
# 2. Google. (n.d.). Case Study 1: How Does a Bike-Share Navigate Speedy Success? [MOOC Lecture]. In Google Data Analytics Capstone: Complete a Case Study. Coursera. https://www.coursera.org/learn/google-data-analytics-capstone
