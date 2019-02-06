import dash
import dash_core_components as dcc
import dash_html_components as html
from .data_structures import *
import plotly.graph_objs as go
from pony.orm import *
from .database import*
from plotly import*
from datetime import datetime
	
def plot_new(measurement_to_plot, dim_2 = False):
	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
	app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
	layout = {}
	figure = {}
	layout['height'] = 1000
	layout['annotations'] = []
	#layout['width'] = 2000
	figure['data'] = []
	if dim_2: type = 'heatmap'
	else: type = 'scatter'
	number_of_qubits = len(measurement_to_plot)
	if number_of_qubits < 3: layout['height'] = 1000
	for index, qubit in enumerate(measurement_to_plot.keys()):
		state = measurement_to_plot[qubit]
		#print(qubit, index, str((index + 1)*2))
		for key in state.datasets.keys():
			number_of_datasets = len(state.datasets.keys())
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
	figure['layout'] = layout
	#db = database()
	#ids = []
	#for qubit in measurement_to_plot.keys(): 
		#for index, type_r in measurement_to_plot[qubit].references:
			#ids.append(index)
	#query = select(i for i in db.Reference if (i.that.id in ids))
	#references = {}
	#for q in query: references.update({q.this.id: q.ref_rype})
	app.layout = html.Div(children=[
		html.H1(state.measurement_type, style={'textAlign': 'center', 'fontSize': 35}),
		dcc.Graph(figure = figure),
		html.Div(html.P('Started at: ' + str(state.start)), style={'fontSize': 18}),
		html.Div(html.P('Stopped at: ' + str(state.stop)), style={'fontSize': 18}),
		html.Div(html.P('Owner: ' + str(state.owner)), style={'fontSize': 18}),
		html.Div(html.P('Metadata: ' + str(state.metadata)), style={'fontSize': 18}),
		html.Div(html.P('References: ' + str(state.references)), style = {'frontSize': 18}),
		#(html.Div(a) for a in state.metadata.items())
	])
	app.run_server(debug=False)