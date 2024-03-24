import googlemaps
import requests
import json
import time
from csv import writer
import pandas as pd
import pause
import datetime
import asyncio
from asgiref.sync import sync_to_async

API_key = "??????"
url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial"
result_list = dict()
api_call = dict()

# function to add new row in the result csv file
def append_list_as_row(file_name, list_of_elem):
	with open(file_name, 'a', newline='') as write_obj:
		csv_writer = writer(write_obj)
		csv_writer.writerow(list_of_elem)

def prepare_apis(idx, source_lat, source_long, dest_lat, dest_long):
	originPoint = source_lat + "," + source_long
	destinationPoint = dest_lat + "," + dest_long
	api_call[idx] = url + "&origins=" + originPoint + "&destinations=" + destinationPoint + "&departure_time=now" + "&key=" + API_key
	print('API', idx, api_call[idx])

# function calls the api for source and destination and returns the api return parameters in the form of a list
def get_api_request_list(idx):
	query_time1 = time.time()
	res1 = json.dumps({'aa':1}) if 0 else requests.get(api_call[idx]).json()
	query_time2 = time.time()
	try:
		res = res1['rows'][0]['elements'][0]
		distance_text = res['distance']['text']
		distance_value = res['distance']['value']
		duration_text = res['duration']['text']
		duration_value = res['duration']['value']
		duration_in_traffic_text = res['duration_in_traffic']['text']
		duration_in_traffic_value = res['duration_in_traffic']['value']
		result_list[idx] = [idx, datetime.datetime.now(), time.ctime(query_time1), round(query_time2-query_time1,2), distance_text, distance_value, duration_text, duration_value, duration_in_traffic_text, duration_in_traffic_value]
	except Exception as e:
		#print('traceback.format_exc():\n%s' % traceback.format_exc())
		print(res1)
		result_list[idx] = [idx, datetime.datetime.now(), -1, round(query_time2-query_time1,2), -1, -1, -1, -1, -1, -1]	
	return idx

async def call_1api(j):
	await sync_to_async(get_api_request_list)(j)

async def call_apis(coordinates):
    await asyncio.gather(*[call_1api(coordinates[0][j]+'-1') for j in range(len(coordinates))])

async def call_apis2(coordinates):
    await asyncio.gather(*[call_1api(coordinates[0][j]+'-2') for j in range(len(coordinates))])


# reads space separated latitude longitude coordinates from a file and passes them as string to get_api_request_list func
def main():

	coordinates = pd.read_csv('final_sub_routes.csv', header=None, skiprows=1, nrows=1, converters={i: str for i in range(5)})
	print(coordinates)
	cnt = len(coordinates[0])
	print('cnt', cnt)
	for j in range(cnt):
		prepare_apis(coordinates[0][j]+'-1', coordinates[1][j], coordinates[2][j], coordinates[3][j], coordinates[4][j])
		prepare_apis(coordinates[0][j]+'-2', coordinates[3][j], coordinates[4][j], coordinates[1][j], coordinates[2][j])

	loop = asyncio.get_event_loop()
	month = 3
	tm_scale = 30*60
	for dt in range(12,12+7):

		tm = datetime.datetime(2021, month, dt, 6)
		tm = datetime.datetime.now()
		add = datetime.timedelta(0,tm_scale)
		for i in range(0, (16*60*60)//tm_scale):
			pause.until(tm)
			tm += add
			print(i, datetime.datetime.now())

			loop.run_until_complete(call_apis(coordinates))
			loop.run_until_complete(call_apis2(coordinates))

			for j in range(cnt):
				print(result_list[coordinates[0][j]+'-1'])
				append_list_as_row('PollGTime'+str(month)+str(dt)+'.csv', result_list[coordinates[0][j]+'-1'])
				append_list_as_row('PollGTime'+str(month)+str(dt)+'.csv', result_list[coordinates[0][j]+'-2'])

	loop.close()

if __name__ == '__main__':
	main()

