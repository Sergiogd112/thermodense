```mermaid
flowchart TD
    tudelft_data[TU Delft Data]; 
    msis[MSIS Models]; 
    fetch_space_weather[Fetch Space Weather Data]; 
    timestamp[Timestamp]
    tudelft_data --- timestamp
    timestamp --> fetch_space_weather
    timestamp --> msis
```
