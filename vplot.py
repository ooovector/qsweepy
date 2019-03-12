from qsweepy import*
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np

#import exdir
#from data_structures import *
import plotly.graph_objs as go
from pony.orm import *
#from database import database
from plotly import*
#from datetime import datetime
from dash.dependencies import Input, Output
import pandas as pd
#import logging

#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
db = database.database()

def string_to_list(string):
	if string == '': return []
	position = string.find(',')
	list_res = []
	while position > -1:
		#print(position, 'and', string[:10], 'and', string[position+1:])
		list_res.append(string[:position])
		string = string[position+2:]
		position = string.find(',')
	if string != '': list_res.append(string)
	return list_res

def data_to_dict(data):
		return { 'id': data.id,
				 #comment': data.comment,
				 'sample_name': data.sample_name,
				'time_start': data.time_start,
				 'time_stop': data.time_stop,
				 #'filename': data.filename,
				 'type_revision': data.type_revision,
				 'incomplete': data.incomplete,
				 'invalid': data.invalid,
				 'owner': data.owner,
				} 

def generate_table(dataframe, max_rows=10):
	return html.Table(
			# Header
			[html.Tr([html.Th(col) for col in dataframe.columns])] +

			# Body
			[html.Tr([
				html.Td(str(dataframe.iloc[i][col])) for col in dataframe.columns
			]) for i in range(min(len(dataframe), max_rows))])

def plot_from_file(meas_ids, fit_ids = '', dim_2 = False): ### TODO: rather useless way of setting dim_2 right now
	layout = {}
	figure = {}
	#print(db.Data[1].filename)
	app.layout = html.Div(children=[
			html.Div([
				html.H1(id = 'list_of_meas_types', style = {'fontSize': '30', 'text-align': 'left', 'text-indent': '5em'}),
				dcc.Graph(id = 'live-plot-these-measurements')], style = {'position': 'absolute', 'top': '30', 'left': '30', 'width': '2200' , 'height': '1200'}),
			html.Div([
				#html.H2(children = 'Measurements', style={'fontSize': 25}),
				html.Div(id = 'table_of_meas'),
				#html.Div(html.P('Started at: ' + str(start)), style={'fontSize': 14}),
				#html.Div(html.P('Stopped at: ' + str(stop)), style={'fontSize': 14}),
				#html.Div(html.P('Owner: ' + str(owner)), style={'fontSize': 14}),
				#html.Div(html.P('Metadata: ' + met_arr), style={'fontSize': 14}),
				html.H3(children = 'Measurement info', style={'fontSize': 25}),
				html.Div(id = 'dropdown', style = {'width': '100'}),
				html.Div(id = 'meas_info'),
				html.Div(html.P('Reference is below')),
				html.Div([html.P('You chose following measurements: '), dcc.Input(id='meas-id', value = str(meas_ids))]), 
				html.Div([html.P('You chose following fits: '), dcc.Input(id='fit-id', value = str(fit_ids), type = 'string')])],
						 style={'position': 'absolute', 'top': '80', 'left': '1500', 'width': '350' , 'height': '800',
								'padding': '0px 10px 15px 10px',
								  'marginLeft': 'auto', 'marginRight': 'auto', #'background': 'rgba(167, 232, 170, 1)',
								'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'},#rgba(190, 230, 192, 1)'},
			   ),
				#html.Div([html.Div(id='my-div', style={'fontSize': 14}), dcc.Input(id='meas-id', value = str(meas_ids), type='string')])]),
			html.Div(id='intermediate-value-meas', style={'display': 'none'}),
			html.Div(id='intermediate-value-fit', style={'display': 'none'}),
			#dcc.Input(id='meas-id', value = str(meas_ids), style={}),
			html.Div([html.H4(children='References', style={'fontSize': 25}), html.Div(id = 'table_of_references')], style = {'position': 'absolute', 'top': '1100', 'left': '50'})
			#(html.Div(a) for a in state.metadata.items())
	])
	
	@app.callback(
		Output(component_id = 'meas_info', component_property = 'children'),
		[Input(component_id = 'my_dropdown', component_property='value')])
	def write_meas_info(value):
		with db_session:
			if value == None: return 
			state = save_exdir.load_exdir(db.Data[int(value)].filename ,db)
			met_arr = ''
			for k in state.metadata:
				if k[0] != 'parameter values': met_arr += str(k[0]) + ' = ' + str(k[1]) + '\n '
			#print(met_arr)
			return (html.Div(html.P('Started at: ' + str(state.start)), style={'fontSize': 14}),
							html.Div(html.P('Stopped at: ' + str(state.stop)), style={'fontSize': 14}),
							html.Div(html.P('Owner: ' + str(state.owner)), style={'fontSize': 14}),
							html.Div(html.P('Metadata: ' + met_arr), style={'fontSize': 14}))
						
	@app.callback(
		Output(component_id = 'dropdown', component_property = 'children'),
		[Input(component_id = 'meas-id', component_property='value')])
	def create_dropdown(meas_ids):
		return dcc.Dropdown(id = 'my_dropdown', options = [{'label': str(i), 'value': str(i)} for i in string_to_list(meas_ids)])
	
	#@app.callback(
		#Output(component_id = 'table_of_meas', component_property = 'children'),
		#[Input(component_id = 'meas-id', component_property='value')])
	def generate_table_of_meas(meas_ids):
		with db_session:
			ids = []
			type_ref = []
			meas_accordance_to_references = []
			df = pd.DataFrame()
			if meas_ids != '':
				for i in string_to_list(meas_ids):
					state = save_exdir.load_exdir(db.Data[int(i)].filename, db)
					df = df.append({'id': state.id, 'owner': state.owner}, ignore_index=True)
			return generate_table(df)
	
	@app.callback(
		Output(component_id = 'table_of_references', component_property = 'children'),
		[Input(component_id = 'meas-id', component_property='value')])
	def generate_table_of_references(meas_ids):
		with db_session:
			meas_accordance_to_references = []
			df = pd.DataFrame()
			if meas_ids != '':
				for i in string_to_list(meas_ids):
					ids = []
					type_ref = []
					state_references = save_exdir.load_exdir(db.Data[int(i)].filename, db).references.items()
					for index, type_r in state_references:
						ids.append(index)
						#id_keys.update({index: i})
						type_ref.append(type_r)
					query_for_table = select(c for c in db.Data if (c.id in ids))
					df_new = pd.DataFrame()
					df_new = pd.DataFrame(list(data_to_dict(x) for x in list(query_for_table)))
					if not df_new.empty:
						df_new = df_new.assign(reference_type=pd.Series(type_ref), ignore_index=True)
						df_new = df_new.assign(measurement = pd.Series(np.full(len(ids), i)), ignore_index=True)
						df = df.append(df_new, ignore_index=True)
						#for i in df_new['id']:
							#meas_accordance_to_references.append(id_keys.get(i))
			#if not df.empty:
				#df = df.assign(reference_type=pd.Series(type_ref), ignore_index=True)
				#df = df.assign(measurement = pd.Series(np.full(len(ids), i)), ignore_index=True)
				#df = df.assign(measurement = pd.Series(meas_accordance_to_references), ignore_index=True)
			#print(df)
			return generate_table(df)
	@app.callback(
		Output(component_id = 'list_of_meas_types', component_property = 'children'),
		[Input(component_id = 'meas-id', component_property='value')])
	def add_meas_type(input_value):
		with db_session:
			list_of_states = string_to_list(input_value)
			list_of_meas_types = []
			for i in list_of_states:
				if (save_exdir.load_exdir(db.Data[int(i)].filename, db)).measurement_type not in list_of_meas_types:
					if list_of_meas_types != []: list_of_meas_types.append(', ') 
					list_of_meas_types.append((save_exdir.load_exdir(db.Data[int(i)].filename)).measurement_type) 
			return list_of_meas_types
				
	@app.callback(
		Output(component_id='intermediate-value-meas', component_property='children'),
		[Input(component_id='meas-id', component_property='value')])

	def add_meas(input_value):
		return string_to_list(str(input_value))
		
	@app.callback(
		Output(component_id='intermediate-value-fit', component_property='children'),
		[Input(component_id='fit-id', component_property='value')])

	def add_meas2(input_value):
		return string_to_list(str(input_value))

	@app.callback(Output('live-plot-these-measurements', 'figure'),
				  [Input('intermediate-value-fit', 'children'), Input('intermediate-value-meas', 'children')])
	def plot_these_measurements(fit_ids_saved, meas_ids_saved): 
		with db_session:
			#print(measurement_to_plot)
			#print(meas_ids_saved)
			measurement_to_fit = {}
			measurement_to_plot = {}
			if meas_ids_saved != '':
				for index, i in enumerate(meas_ids_saved):
					state = save_exdir.load_exdir(db.Data[int(i)].filename, db)
					measurement_to_plot.update({str(index):state})#({db.Data[int(i)].sample_name: state})
			if fit_ids_saved != '':
				for index, i in enumerate(fit_ids_saved):
					state = save_exdir.load_exdir(db.Data[int(i)].filename, db)
					measurement_to_fit.update({str(index): state})#({db.Data[int(i)].sample_name: state})

			#print('I am working ', n_intervals)
			layout['height'] = 1000
			layout['annotations'] = []
			layout['width'] = 1500
			layout['showlegend'] = False
			figure['data'] = []
			if dim_2: type = 'heatmap' ### 
			else: type = 'scatter'
			number_of_qubits = len(measurement_to_plot)
			if number_of_qubits < 3: layout['height'] = 900
			for qubit_index, qubit in enumerate(measurement_to_plot.keys()):
				state = measurement_to_plot[qubit]
				for i, key in enumerate(state.datasets.keys()):
					number_of_datasets = len(state.datasets.keys())
					#print(state.datasets[key].parameters[0].values, state.datasets[key].parameters[1].values, state.datasets[key].data)
					if (number_of_datasets == 1) and (number_of_qubits < 3): layout['width'] = 1000
					number = number_of_qubits*number_of_datasets
					index = i + qubit_index
					layout['xaxis' + str((index + 1)*2)] = {'anchor': 'x' + str((index + 1)*2), 'domain': [index/number, (index + 0.8)/number], 
										'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #RE
					layout['yaxis' + str((index + 1)*2)] = {'anchor': 'x' + str((index + 1)*2), 'domain': [0, 0.45], 
										'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
					layout['xaxis' + str((index + 1)*2 + 1)] = {'anchor': 'x' + str((index + 1)*2 + 1), 'domain': [index/number, (index + 0.8)/number],
										'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #IM
					layout['yaxis' + str((index + 1)*2 + 1)] = {'anchor': 'x' + str((index + 1)*2 + 1), 'domain': [0.55, 1], 
										'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
					dataset = state.datasets[key]
					figure['data'].append({'colorbar': {'len': 0.4,
										   'thickness': 0.025,
										   'thicknessmode': 'fraction',
										   'x': (index + 0.8)/number,
										   'y': 0.2},
							  'type': type,
							  'mode': 'markers' if not dim_2 else '',
							  'uid': '',
							  'xaxis': 'x' + str((index + 1)*2),
							  'yaxis': 'y' + str((index + 1)*2),
							  'x': np.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
							  'y': np.memmap.tolist(np.imag(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
							  'z': np.memmap.tolist(np.imag(dataset.data))})
					layout['annotations'].append({'font': {'size': 16},
										'showarrow': False,
										'text': str(state.id) + ': Re(' + key + ')',
										'x': (index + 0.4)/number,
										'xanchor': 'center',
										'xref': 'paper',
										'y': 1,
										'yanchor': 'bottom', 'yref': 'paper'})
					figure['data'].append({'colorbar': {'len': 0.4,
										   'thickness': 0.025,
										   'thicknessmode': 'fraction',
										   'x': (index + 0.8)/number,
										   'y': 0.8},
							  'type': type,
							  'mode': 'markers' if not dim_2 else '',
							  'uid': '',
							  'xaxis': 'x' + str((index + 1)*2 + 1),
							  'yaxis': 'y' + str((index + 1)*2 + 1),
							  'x': np.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
							  'y': np.memmap.tolist(np.real(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
							  'z': np.memmap.tolist(np.real(dataset.data))})
					layout['annotations'].append({'font': {'size': 16},
										'showarrow': False,
										'text': str(state.id) + ': Im(' + key + ')',
										'x': (index + 0.4)/number,
										'xanchor': 'center',
										'xref': 'paper',
										'y': 0.45,
										'yanchor': 'bottom', 'yref': 'paper'})  
					if (len(fit_ids_saved) > 0) and (qubit in measurement_to_fit.keys()):
						fit_state = measurement_to_fit[qubit]
						for key in fit_state.datasets.keys(): 
							figure['data'].append({'colorbar': {'len': 0.4,
											   'thickness': 0.025,
											   'thicknessmode': 'fraction',
												'x': (index + 0.8)/number,
											   'y': 0.2},
								  'type': type,
								  'mode': 'lines' if not dim_2 else '',
								  'uid': '',
								  'xaxis': 'x' + str((index + 1)*2),
								  'yaxis': 'y' + str((index + 1)*2),
								  'x': np.memmap.tolist(np.real(fit_state.datasets[key].parameters[0].values)),
								  'y': np.memmap.tolist(np.imag(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(fit_state.datasets[key].parameters[1].values)),
								  'z': np.memmap.tolist(np.imag(dataset.data))})
							figure['data'].append({'colorbar': {'len': 0.4,
											   'thickness': 0.025,
											   'thicknessmode': 'fraction',
											   'x': (index + 0.8)/number,
											   'y': 0.8},
								  'type': type,
								  'mode': 'lines' if not dim_2 else '',
								  'uid': '',
								  'xaxis': 'x' + str((index + 1)*2 + 1),
								  'yaxis': 'y' + str((index + 1)*2 + 1),
								  'x': np.memmap.tolist(np.real(fit_state.datasets[key].parameters[0].values)),
								  'y': np.memmap.tolist(np.real(dataset.data)) if not dim_2 else np.memmap.tolist(np.real(fit_state.datasets[key].parameters[1].values)),
								  'z': np.memmap.tolist(np.real(dataset.data))})
			figure['layout'] = layout
			return figure
		
if __name__ == '__main__':
	plot_from_file('')
	app.run_server(debug=False)