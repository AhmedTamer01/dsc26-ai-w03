# AI usage



## Tools I used, and for what

I used ChatGpt to understand the Requrements more and also review my implmentation with also helping me remembering some of the built in functions from the libraries

## AI-suggested code I rejected or fixed in add_history_features or time_split

The initial history feature approach using cumcount() and transform(mean) was incorrect because it could count parcels with the same timestamp as prior parcels and include future failure outcomes I changed the implementation to aggregate by customer and timestamp then used shifted cumulative values I also replaced merge() with join() to preserve the original Dataframe index and row order

## How I checked that my model has no leak

I selected only features available when a parcel is created excluded post delivery columns such as courier_notes and delivered_at and used a time based train test split
