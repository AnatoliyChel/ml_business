# USAGE
# Start the server:
# 	python run_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import dill
import pandas as pd
dill._dill._reverse_typemap['ClassType'] = type
# import cloudpickle
import flask
import datetime


# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
	return "Welcome to fraudelent prediction process"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		request_json = flask.request.get_json()
		wsbp, wsrh, wsth, date = request_json.values()

		# convert input data to month number
		d = datetime.datetime.strptime(date, '%Y-%m-%d')
		month = d.strftime("%m").lstrip("0").replace(" 0", " ")
		key_month = "month_" + str(month)

		# convert input data to day number
		day = d.strftime("%d").lstrip("0").replace(" 0", " ")
		key_day = "day_" + str(day)

		# create Test_dataset
		col_months = ["month_" + str(i) for i in range(1, 13)] # column names into Test dataset
		col_days = ["day_" + str(i) for i in range(1, 32)]  # column names into Test dataset

		df_months_days = pd.DataFrame(0, index=range(1), columns=col_months + col_days)
		df_months_days.loc[0, [key_month, key_day]] = 1
		final_df = pd.DataFrame({"wsbp.csv": [wsbp], "wsrh.csv": [wsrh], "wsth.csv": [wsth]})
		final_df = final_df.merge(df_months_days, left_index=True, right_index=True)

		# get predicts
		predicts = model.predict(final_df)

		# create output message
		data["predictions"] = str(predicts[0])
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	modelpath = "xgboost_model.dill"
	load_model(modelpath)
	app.run()