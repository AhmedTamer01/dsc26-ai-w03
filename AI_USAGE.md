# AI usage



## Tools I used, and for what

I used ChatGpt to understand the Requrements more and also review my implmentation with also helping me remembering some of the built in functions from the libraries.
also used claude to adjust some functions after i finished the task that helped me avoid some edge cases i couldnt notice

## AI-suggested code I rejected or fixed in add_history_features or time_split

The Ai Gave me a history feature approach in add_history_features using transform(mean) and it was incorrect because it counted the parcels within the same timestamp and that was wrong i caught that with trying 2 parcels at the same time for the first check

## How I checked that my model has no leak

I selected only features available when a parcel is created excluded post delivery columns such as courier_notes and delivered_at and used a time based train test split.
