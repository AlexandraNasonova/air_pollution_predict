#!/bin/bash

echo "*** Updating date in params.yaml file..."

# Get the current date
new_date=$(date -d "yesterday" +%Y-%m-%d)
# Get the previous date
old_date=$(grep -m 1 "current_date" params.yaml | cut -d "'" -f 2)

# Save the previous current_date value to last_run_date
sed -i "s/\(last_run_date: \&last_run_date '\)[^']*\('.*\)/\1$old_date\2/g" params.yaml

# Update the current_date value
sed -i "s/\(current_date: \&current_date '\)[^']*\('.*\)/\1$new_date\2/g" params.yaml

# Update the date_end_val value
sed -i "s/\(date_end_val: \&date_end_val '\)[^']*\('.*\)/\1$new_date\2/g" params.yaml

# Run pythons scripts
echo "*** Dates are up to date"

echo "*** Downloading, clean, enrich etc. actual source data..."

python3 download_weather.py --output_file datasets/weather-source-data/weather_cur_year.csv  --params params.yaml --params_section download-weather-cur-year
python3 download_pollutants.py --output_folder datasets/pollutants-source-data/cur_year --params params.yaml --params_section download-pollutants-cur-year
python3 clean_pollutants.py --input_folder datasets/pollutants-source-data/cur_year/ --output_folder datasets/pollutants-clean-data/cur_year/ --params params.yaml
python3 enrich_pollutants.py --input_cur_year_folder datasets/pollutants-clean-data/cur_year/ --input_prev_years_folder datasets/pollutants-clean-data/prev_years/ --output_file datasets/pollutants-enrich-data/aqi_enriched_cur_year.csv --params params.yaml --params_section enrich-pollutants-cur-year
python3 clean_weather.py --input_file datasets/weather-source-data/weather_cur_year.csv --output_file datasets/weather-clean-data/weather_cur_year.csv --params params.yaml
python3 merge_enriched_weather.py --input_weather_prev_years_file datasets/weather-clean-data/weather_prev_years.csv --input_weather_cur_year_file datasets/weather-clean-data/weather_cur_year.csv --input_aqi_prev_years_file datasets/pollutants-enrich-data/aqi_enriched_prev_years.csv --input_aqi_cur_year_file datasets/pollutants-enrich-data/aqi_enriched_cur_year.csv --output_file datasets/pollutants-weather-merged-data/aqi_all.csv --params params.yaml

echo "*** Source data are up to date"
echo "*** Retraining model with actual data..."

python3 filter_columns_for_model.py --input_file datasets/pollutants-weather-merged-data/aqi_all.csv --output_file experiments_results/lgbm/6001/aqi_filtered.csv --params params.yaml --params_section lgbm_pm25
python3 split_train_val.py --input_file experiments_results/lgbm/6001/aqi_filtered.csv --output_train_file experiments_results/lgbm/6001/train.csv --output_val_file experiments_results/lgbm/6001/val.csv --params params.yaml
python3 train_lgbm_model.py --input_train_file experiments_results/lgbm/6001/train.csv --input_val_file experiments_results/lgbm/6001/val.csv --input_model_params_file experiments_results/lgbm/6001/model_params.json --output_metrics_file experiments_results/lgbm/6001/metrics.json --output_onnx_file experiments_results/lgbm/6001/model.onnx --params params.yaml --params_section lgbm_pm25

echo "*** Model is up to date"

echo "*** SUCCESS. Data and model update is finished"