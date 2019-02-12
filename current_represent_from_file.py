import dash
import dash_core_components as dcc
import dash_html_components as html
from .data_structures import *
import plotly.graph_objs as go
from pony.orm import *
from .database import database
from plotly import*
from datetime import datetime
from dash.dependencies import Input, Output
import pandas as pd
from .save_exdir import*
import ast
import json

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
				 'comment': data.comment,
				 'sample_name': data.sample_name,
				 'time_start': data.time_start,
				 'time_stop': data.time_stop,
				 'filename': data.filename,
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
			]) for i in range(min(len(dataframe), max_rows))]
		)
	
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True	

#measurement_type, start, stop, owner, met_arr, df = return_text(measurement_to_plot)

def plot_from_file(meas_ids, fit_ids = '', dim_2 = False):
	layout = {}
	figure = {}
	db = database()
	measurement_to_plot = {}
	measurement_to_fit = {}
	for i in string_to_list(meas_ids):
		#print(db.Data[int(i)])
		state = read_exdir_new(db.Data[int(i)].filename)
		measurement_to_plot.update({db.Data[int(i)].sample_name: state})
	
	
	def return_text(measurement_to_plot):
		number_of_qubits = len(measurement_to_plot) - 1
		db = database()
		ids = []
		type_ref = []
		for qubit in measurement_to_plot.keys(): 
			state = measurement_to_plot[qubit]
			for index, type_r in measurement_to_plot[qubit].references.items():
				ids.append(index)
				type_ref.append(type_r)
		query_for_table = select(c for c in db.Data if (c.id in ids))
		df = pd.DataFrame(list(data_to_dict(x) for x in list(query_for_table)))
		#for q in query_for_table: 
			#type_ref.append(get(i.ref_type for i in db.Reference if (i.that.id == q.id)))
		#print(type_ref)
		df = df.assign(reference_type=pd.Series(type_ref))
		met_arr = ''
		for k in state.metadata:
			if k[0] != 'parameter values': met_arr += str(k[0]) + ' = ' + str(k[1]) + '\n '
		return state.measurement_type, state.start, state.stop, state.owner, met_arr, df 
		
	#print(measurement_to_plot['1'].references)
	measurement_type, start, stop, owner, met_arr, df = return_text(measurement_to_plot)
	
	app.layout = html.Div(children=[
			html.Div([
				html.H1(measurement_type, style = {'fontSize': '30', 'text-align': 'left', 'text-indent': '5em'}),
				dcc.Graph(id = 'live-plot-these-measurements')], style = {'position': 'absolute', 'top': '30', 'left': '30', 'width': '2200' , 'height': '1200'}),
			html.Div([
				html.H2(children = 'Measurement info', style={'fontSize': 25}),
				html.Div(html.P('Started at: ' + str(start)), style={'fontSize': 14}),
				html.Div(html.P('Stopped at: ' + str(stop)), style={'fontSize': 14}),
				html.Div(html.P('Owner: ' + str(owner)), style={'fontSize': 14}),
				html.Div(html.P('Metadata: ' + met_arr), style={'fontSize': 14}),
				html.Div(html.P('Reference is below')),
				html.Div([html.P('You chose following measurements: '), dcc.Input(id='meas-id', value = str(meas_ids))]), 
				html.Div([html.P('You chose following fits: '), dcc.Input(id='fit-id', value = str(fit_ids), type = 'string')])],
				style = {'position': 'absolute', 'top': '80', 'left': '1500', 'width': '400' , 'height': '700'}),
				#html.Div([html.Div(id='my-div', style={'fontSize': 14}), dcc.Input(id='meas-id', value = str(meas_ids), type='string')])]),
			html.Div(id='intermediate-value-meas', style={'display': 'none'}),
			html.Div(id='intermediate-value-fit', style={'display': 'none'}),
			#dcc.Input(id='meas-id', value = str(meas_ids), style={}),
			html.Div([html.H3(children='References', style={'fontSize': 25}), generate_table(df)], style = {'position': 'absolute', 'top': '1100', 'left': '50'})
			#(html.Div(a) for a in state.metadata.items())
	])
	#app.run_server(debug=False)
	
	#@app.callback(Output('intermediate-value-fit', 'children'), [Input('my-id', 'number')])
	
	#def add_filename_meas(id):
		#if id == None: return 
		#db = database()
		#filename = db.Data[id].filename
		#measurement_to_fit_filenames += filename
		#return measurement_to_fit_filename
		
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
	
	@app.callback(Output('live-plot-these-measurements', 'figure'),[Input('intermediate-value-fit', 'children'), Input('intermediate-value-meas', 'children')])

	def plot_these_measurements(fit_ids_saved, meas_ids_saved): 
		#print(measurement_to_plot)
		#print(meas_ids_saved)
		measurement_to_fit = {}
		measurement_to_plot = {}
		for index, i in enumerate(meas_ids_saved):
			#print(i)
			state = read_exdir_new(db.Data[int(i)].filename)
			measurement_to_plot.update({str(index):state})#({db.Data[int(i)].sample_name: state})
		if fit_ids_saved != '':
			for index, i in enumerate(fit_ids_saved):
				state = read_exdir_new(db.Data[int(i)].filename)
				measurement_to_fit.update({str(index): state})#({db.Data[int(i)].sample_name: state})
		
		#print('I am working ', n_intervals)
		layout['height'] = 1000
		layout['annotations'] = []
		layout['width'] = 1500
		layout['showlegend'] = False
		figure['data'] = []
		if dim_2: type = 'heatmap'
		else: type = 'scatter'
		number_of_qubits = len(measurement_to_plot)
		if number_of_qubits < 3: layout['height'] = 900
		for qubit_index, qubit in enumerate(measurement_to_plot.keys()):
			state = measurement_to_plot[qubit]
			#print(measurement_to_fit)
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
						  'x': numpy.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
						  'y': numpy.memmap.tolist(np.imag(dataset.data)) if not dim_2 else numpy.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
						  'z': numpy.memmap.tolist(np.imag(dataset.data))})
				layout['annotations'].append({'font': {'size': 16},
									'showarrow': False,
									'text': qubit + ': Re(' + key + ')',
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
						  'x': numpy.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
						  'y': numpy.memmap.tolist(np.real(dataset.data)) if not dim_2 else numpy.memmap.tolist(np.real(state.datasets[key].parameters[1].values)),
						  'z': numpy.memmap.tolist(np.real(dataset.data))})
				layout['annotations'].append({'font': {'size': 16},
									'showarrow': False,
									'text': qubit + ': Im(' + key + ')',
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
							  'x': numpy.memmap.tolist(np.real(fit_state.datasets[key].parameters[0].values)),
							  'y': numpy.memmap.tolist(np.imag(dataset.data)) if not dim_2 else numpy.memmap.tolist(np.real(fit_state.datasets[key].parameters[1].values)),
							  'z': numpy.memmap.tolist(np.imag(dataset.data))})
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
							  'x': numpy.memmap.tolist(np.real(fit_state.datasets[key].parameters[0].values)),
							  'y': numpy.memmap.tolist(np.real(dataset.data)) if not dim_2 else numpy.memmap.tolist(np.real(fit_state.datasets[key].parameters[1].values)),
							  'z': numpy.memmap.tolist(np.real(dataset.data))})
		figure['layout'] = layout
		#print(figure)
		return figure
	#plot_update()
	app.run_server(debug=False)
	#measurement_type, start, stop, owner, met_arr, df
def add_to_plot(self, id, measurement_fit = None):
		if measurement_fit == None:
			db = database()
			state = read_exdir_new(db.Data[int(id)].filename)
		else: state = measurement_fit
		type = 'scatter'
		for key in state.datasets.keys():
			number_of_datasets = len(state.datasets.keys())
			number = number_of_datasets
			index = 1
			dataset = state.datasets[key]
			self.figure['data'].append({'colorbar': {'len': 0.4,
							   'thickness': 0.025,
							   'thicknessmode': 'fraction',
							   'x': (index + 0.8)/number,
							   'y': 0.2},
				  'type': type,
				  'mode': 'lines',
				  'uid': '',
				  'xaxis': 'x' + str((index + 1)*2),
				  'yaxis': 'y' + str((index + 1)*2),
				  'x': numpy.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
				  'y': numpy.memmap.tolist(np.imag(dataset.data))})
			self.figure['data'].append({'colorbar': {'len': 0.4,
							   'thickness': 0.025,
							   'thicknessmode': 'fraction',
							   'x': (index + 0.8)/number,
							   'y': 0.2},
				  'type': type,
				  'mode': 'lines',
				  'uid': '',
				  'xaxis': 'x' + str((index + 1)*2 + 1),
				  'yaxis': 'y' + str((index + 1)*2 + 1),
				  'x': numpy.memmap.tolist(np.real(state.datasets[key].parameters[0].values)),
				  'y': numpy.memmap.tolist(np.real(dataset.data))})
		print(dcc.Graph(figure = figure))	
		return figure
	#if __name__ == '__main__':
		#app.run_server(debug=True)
		