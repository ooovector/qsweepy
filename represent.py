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
layout = {}
figure = {}

def plot_new(measurement_to_plot, dim_2 = False):
	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
	app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
	layout = {}
	figure = {}
	layout['height'] = 1000
	layout['annotations'] = []
	layout['width'] = 1500
	figure['data'] = []
	if dim_2: type = 'heatmap'
	else: type = 'scatter'
	number_of_qubits = len(measurement_to_plot)
	if number_of_qubits < 3: layout['height'] = 900
	for index, qubit in enumerate(measurement_to_plot.keys()):
		state = measurement_to_plot[qubit]
		#print(qubit, index, str((index + 1)*2))
		for key in state.datasets.keys():
			number_of_datasets = len(state.datasets.keys())
			if (number_of_datasets == 1) and (number_of_qubits < 3): layout['width'] = 1000
			number = number_of_qubits*number_of_datasets
			layout['xaxis' + str((index + 1)*2)] = {'anchor': 'x' + str((index + 1)*2), 'domain': [index/number, (index + 0.8)/number], 
 							'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #RE
			layout['yaxis' + str((index + 1)*2)] = {'anchor': 'y' + str((index + 1)*2), 'domain': [0, 0.4],
 							'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
			layout['xaxis' + str((index + 1)*2 + 1)] = {'anchor': 'x' + str((index + 1)*2 + 1), 'domain': [index/number, (index + 0.8)/number],
 							'title': state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit} #IM
			layout['yaxis' + str((index + 1)*2 + 1)] = {'anchor': 'y' + str((index + 1)*2 + 1), 'domain': [0.6, 1],
 							'title': state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit if dim_2 else ''}
			dataset = state.datasets[key]
			#print(dataset.parameters)
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
				  'x': np.real(state.datasets[key].parameters[0].values),
				  'y': np.imag(dataset.data) if not dim_2 else np.real(state.datasets[key].parameters[1].values),
				  'z': np.imag(dataset.data)})
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
				  'x': np.real(state.datasets[key].parameters[0].values),
				  'y': np.real(dataset.data) if not dim_2 else np.real(state.datasets[key].parameters[1].values),
				  'z': np.real(dataset.data)})
			layout['annotations'].append({'font': {'size': 16},
							'showarrow': False,
							'text': qubit + ': Im(' + key + ')',
							'x': (index + 0.4)/number,
							'xanchor': 'center',
							'xref': 'paper',
							'y': 0.4,
							'yanchor': 'bottom', 'yref': 'paper'})  
	layout['showlegend'] = False
	figure['layout'] = layout
	db = database()
	ids = []
	type_ref = []
	for qubit in measurement_to_plot.keys(): 
		for index, type_r in measurement_to_plot[qubit].references.items():
			ids.append(index)
			type_ref.append(type_r)
	#query = select(i for i in db.Reference if (i.that.id in ids))
	#references = {}
	#for q in query: references.update({q.this.id: q.ref_rype})
	#print(list(select(c for c in db.Data if (c.id in ids))))
	query_for_table = select(c for c in db.Data if (c.id in ids))
	df = pd.DataFrame(list(data_to_dict(x) for x in list(query_for_table)))
	for q in query_for_table: 
		type_ref = get(i.ref_type for i in db.Reference if (i.that.id == q.id))
	df = df.assign(reference_type=pd.Series(type_ref))
	met_arr = ''
	for k in state.metadata:
		if k[0] != 'parameter values': met_arr += str(k[0]) + ' = ' + str(k[1]) + '\n '
	app.layout = html.Div([
		html.Div([
			html.H1(state.measurement_type, style = {'fontSize': '30', 'text-align': 'left', 'text-indent': '5em'}), #style={'position': 'absolute', 'top': '50', 'left': '150', 'fontSize': 30}),
			dcc.Graph(figure = figure)], style = {'position': 'absolute', 'top': '30', 'left': '30', 'width': '2200' , 'height': '1200'}),#style = {'align': 'left', 'width': '70%', 'display': 'inline-block'})
		html.Div([
			html.H2(children = 'Measurement info', style={'fontSize': 25}),
			html.Div(html.P('Started at: ' + str(state.start)), style={'fontSize': 14}),
			html.Div(html.P('Stopped at: ' + str(state.stop)), style={'fontSize': 14}),
			html.Div(html.P('Owner: ' + str(state.owner)), style={'fontSize': 14}),
			html.Div(html.P('Metadata: ' + str(met_arr)), style={'fontSize': 14}),
			html.Div(html.P('Reference is below')),
			html.Div(html.P('You want to add')),
			html.Div([html.Div(id='my-div', style={'fontSize': '8'}), 
			dcc.Input(id='my-id', value=0, type='number')])], style = {'position': 'absolute', 'top': '80', 'left': '1500', 'width': '400' , 'height': '700'}),#style = {'align': 'right', 'display': 'inline-block', 'width': '20%' , 'height': '1000'}),
		html.Div([html.H3(children='References', style={'fontSize': 25}),
			generate_table(df)], style = {'position': 'absolute', 'top': '1100', 'left': '50'})
			
		#html.Div(html.P('References: ' + str(state.references)), style = {'frontSize': 18}),
		#(html.Div(a) for a in state.metadata.items())
	])
	app.run_server(debug=False)
	
#@app.callback(
#	Output('live-add_to-plot', 'figure'),
#	[Input('interval-component', 'n_intervals')]#[Input(component_id='my-id', component_property='value')])

	
@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)

def update_output_div(input_value):
	return 'You want to add on the page the measurementthe with the following from the Database:  "{}"'.format(input_value)

	
def add_to_plot(id, measurement_fit = None):
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
		figure['data'].append({'colorbar': {'len': 0.4,
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
		figure['data'].append({'colorbar': {'len': 0.4,
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
	