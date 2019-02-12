import dash
import dash_core_components as dcc
import dash_html_components as html
from .data_structures import *
import plotly.graph_objs as go
from pony.orm import *
from .database import*
from plotly import*
from datetime import datetime
from dash.dependencies import Input, Output
import pandas as pd
from .save_exdir import*

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
		
		
		
@app.callback(
	Output(component_id='my-div', component_property='children'),
	[Input(component_id='my-id', component_property='value')])

def update_output_div(input_value):
	return 'You want to add on the page the following measurement from the Database:  "{}"'.format(input_value)
	

	#state=[State('', '')]
		
def plot_new(measurement_to_plot, dim_2 = False):
	layout = {}
	figure = {}
	
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
		for q in query_for_table: 
			type_ref = get(i.ref_type for i in db.Reference if (i.that.id == q.id))
		df = df.assign(reference_type=pd.Series(type_ref))
		met_arr = ''
		for k in state.metadata:
			if k[0] != 'parameter values': met_arr += str(k[0]) + ' = ' + str(k[1]) + '\n '
		return state.measurement_type, state.start, state.stop, state.owner, met_arr, df 
		
		
	measurement_type, start, stop, owner, met_arr, df = return_text(measurement_to_plot)
	
	app.layout = html.Div(children=[
			html.Div([
				html.H1(measurement_type, style = {'fontSize': '30', 'text-align': 'left', 'text-indent': '5em'}),
				dcc.Graph(id = 'live-plot-update')], style = {'position': 'absolute', 'top': '30', 'left': '30', 'width': '2200' , 'height': '1200'}),
			html.Div([
				html.H2(children = 'Measurement info', style={'fontSize': 25}),
				html.Div(html.P('Started at: ' + str(start)), style={'fontSize': 14}),
				html.Div(html.P('Stopped at: ' + str(stop)), style={'fontSize': 14}),
				html.Div(html.P('Owner: ' + str(owner)), style={'fontSize': 14}),
				html.Div(html.P('Metadata: ' + met_arr), style={'fontSize': 14}),
				html.Div(html.P('Reference is below')),
				html.Div([html.Div(id='my-div', style={'fontSize': 14}), dcc.Input(id='my-id', value=0, type='number')])], style = {'position': 'absolute', 'top': '80', 'left': '1500', 'width': '400' , 'height': '700'}),
			dcc.Interval(
				id='interval-component',
				interval=1*1000, # in milliseconds
				n_intervals=0
			),
			html.Div([html.H3(children='References', style={'fontSize': 25}), generate_table(df)], style = {'position': 'absolute', 'top': '1100', 'left': '50'})
			#(html.Div(a) for a in state.metadata.items())
	])
	#app.run_server(debug=False)
	
	@app.callback(Output('live-plot-update', 'figure'),[Input('interval-component', 'n_intervals')])

	def plot_update(n_intervals): 
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
		for index, qubit in enumerate(measurement_to_plot.keys()):
			state = measurement_to_plot[qubit]
			for key in state.datasets.keys():
				number_of_datasets = len(state.datasets.keys())
				if (number_of_datasets == 1) and (number_of_qubits < 3): layout['width'] = 1000
				number = number_of_qubits*number_of_datasets
				layout['xaxis' + str((index + 1)*2)] = {'anchor': 'x' + str((index + 1)*2), 'domain': [index/number, (index + 0.8)/number], 
									'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #RE
				layout['yaxis' + str((index + 1)*2)] = {'anchor': 'y' + str((index + 1)*2), 'domain': [0, 0.45],#0.4
									'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
				layout['xaxis' + str((index + 1)*2 + 1)] = {'anchor': 'x' + str((index + 1)*2 + 1), 'domain': [index/number, (index + 0.8)/number],
									'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #IM
				layout['yaxis' + str((index + 1)*2 + 1)] = {'anchor': 'y' + str((index + 1)*2 + 1), 'domain': [0.55, 1],#0.6
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
						  'y': numpy.memmap.tolist(np.imag(dataset.data) if not dim_2 else np.real(state.datasets[key].parameters[1].values)),
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
						  'y': numpy.memmap.tolist(np.real(dataset.data) if not dim_2 else np.real(state.datasets[key].parameters[1].values)),
						  'z': numpy.memmap.tolist(np.real(dataset.data))})
				layout['annotations'].append({'font': {'size': 16},
									'showarrow': False,
									'text': qubit + ': Im(' + key + ')',
									'x': (index + 0.4)/number,
									'xanchor': 'center',
									'xref': 'paper',
									'y': 0.45,
									'yanchor': 'bottom', 'yref': 'paper'})    
		figure['layout'] = layout
		return figure
	#plot_update()
	app.run_server(debug=False)
	#measurement_type, start, stop, owner, met_arr, df
		
	
