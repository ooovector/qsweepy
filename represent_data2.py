import dash
import dash_core_components as dcc
import dash_html_components as html
from .data_structures import *
import plotly.graph_objs as go
from pony.orm import *
from .database import*
from plotly import*
from datetime import datetime

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
@db_session
def get_databases():
	databases = list(data_to_dict(x) for x in list(select(c for c in db.Data)))
	return pd.DataFrame(databases)

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
	ids = []
	for qubit in measurement_to_plot.keys(): 
		for index, type_r in measurement_to_plot[qubit].references:
			ids.append(index)
	#query = select(i for i in db.Reference if (i.that.id in ids))
	#references = {}
	#for q in query: references.update({q.this.id: q.ref_rype})

	df = pd.Dataframe(list(data_to_dict(x) for x in list(select(c for c in db.Data if (c.id in ids)))))
	app.layout = html.Div(children=[
		html.H1(state.measurement_type, style={'textAlign': 'center', 'fontSize': 35}),
		dcc.Graph(figure = figure),
		html.Div(html.P('Started at: ' + str(state.start)), style={'fontSize': 18}),
		html.Div(html.P('Stopped at: ' + str(state.stop)), style={'fontSize': 18}),
		html.Div(html.P('Owner: ' + str(state.owner)), style={'fontSize': 18}),
		html.Div(html.P('Metadata: ' + str(state.metadata)), style={'fontSize': 18}),
		html.Div(html.P('References: ' + str(state.references)), style = {'frontSize': 18}),
		html.H2(children='References'),
		generate_table(df)
		#(html.Div(a) for a in state.metadata.items())
	])
	app.run_server(debug=False)
	
	
def plot_with_dash_2D(state):
	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
	app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
	trace = []
	num_of_rows = len(state.datasets.keys())
	titles = []
	xaxis = {}
	for key in state.datasets.keys():
		trace_help = []
		#print(np.real(state.datasets[key].data), state.datasets[key].parameters[1].values)
		trace_help.append(go.Heatmap(
			#x = state.datasets[key].parameters[0].values,
            #y = state.datasets[key].parameters[1].values,
            z = np.real(state.datasets[key].data), colorbar = dict(x=0.45)))
		trace_help.append(go.Heatmap(
			#x = state.datasets[key].parameters[0].values,
            #y = state.datasets[key].parameters[1].values,
            z = np.imag(state.datasets[key].data)))
		titles.append('Re(' + str(key) + ')')
		titles.append('Im(' + str(key) + ')')
		trace.append(trace_help)
		#xaxis.update({xaxis: dict(title=state.datasets[key].parameters[1].name)})
		#yaxis.update({yaxis: dict(title=state.datasets[key].parameters[0].name)})
	fig = tools.make_subplots(rows = num_of_rows, cols = 2, subplot_titles = titles)
	yaxis=dict(title=state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit, titlefont=dict(size=16))#, color='black'))
	xaxis=dict(title=state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit, titlefont=dict(size=16))#, color='black'))
	#layout = go.Layout(yaxis1 = yaxis, xaxis1 = xaxis, yaxis2 = yaxis, xaxis2 = xaxis)
	for i in range(num_of_rows):
		for j in range(2):
			fig.append_trace(trace[i][j], i+1, j+1)
	#fig['layout'].update(layout)
	for i in range (num_of_rows*2):
		fig['layout']['xaxis'+str(i+1)].update(xaxis)
		fig['layout']['yaxis'+str(i+1)].update(yaxis)

	app.layout = html.Div(children=[
		html.H1(state.measurement_type, style={'fontSize': 30}),
		dcc.Graph(figure=fig),
		html.Div(html.P('Started at: ' + str(state.start)), style={'fontSize': 18}),
		html.Div(html.P('Stopped at: ' + str(state.stop)), style={'fontSize': 18}),
		html.Div(html.P('Owner: ' + str(state.owner)), style={'fontSize': 18}),
		html.Div(html.P('Metadata: ' + str(state.metadata)), style={'fontSize': 18}),
		#html.Div([a for a in state.metadata.items()])
	])
	app.run_server(debug=False)
	
def plot_with_dash_2D_anticrossings(states, Re, Im):
	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
	app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
	trace = []
	titles = []
	num_of_qubits = len(states)
	num_of_parameters = 0
	for key in states.keys():
		if num_of_parameters < len(states[key].datasets.keys()): num_of_parameters = len(states[key].datasets.keys())
	if Re and Im: num_of_parameters *= 2
		#raise 'Please, select one part of the imaginary number'
	#fig = tools.make_subplots(rows = num_of_qubits, cols = num_of_parameters)
	for index, qubit in enumerate(states.keys()):
		state = states[qubit]
		trace_help = []
		for parameter_id, key in enumerate(state.datasets.keys()):
			#yaxis=dict(title=state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit, titlefont=dict(size=16), domain = [index/num_of_qubits, (index + 0.8)/num_of_qubits])#, color='black'))
			#xaxis=dict(title=state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit, titlefont=dict(size=16), domain = [parameter_id/num_of_parameters, (parameter_id+0.8)/num_of_parameters])#, color='black'))
			if Re:	
				trace_help.append(go.Heatmap(
					#x = state.datasets[key].parameters[0].values,
					#y = state.datasets[key].parameters[1].values,
					z = np.real(state.datasets[key].data),
					colorbar = dict(x = (parameter_id+0.95)/num_of_parameters, y = (index+0.35)/num_of_qubits, len = 0.7/num_of_qubits, thicknessmode = 'fraction', thickness = 0.05/num_of_parameters)))
				titles.append(qubit + ': Re(' + str(key) + ')')
				
			if Im:
				trace_help.append(go.Heatmap(
					#x = state.datasets[key].parameters[0].values,
					#y = state.datasets[key].parameters[1].values,
					z = np.imag(state.datasets[key].data),
					colorbar = dict(x = (parameter_id+1.95)/num_of_parameters, y = (index+0.35)/num_of_qubits, len = 0.7/num_of_qubits, thicknessmode = 'fraction', thickness = 0.05/num_of_parameters)))
				titles.append(qubit + ': Im(' + str(key) + ')')
			
			#print(parameter_id, index, str(parameter_id+1+index*num_of_parameters))
			#fig['layout']['xaxis'+str(parameter_id+1+index*num_of_parameters)].update(xaxis)
			#fig['layout']['yaxis'+str(parameter_id+1+index*num_of_parameters)].update(yaxis)
			
		trace.append(trace_help)
	#fig = tools.make_subplots(rows = num_of_qubits, cols = num_of_parameters, shared_xaxes = True, subplot_titles = titles)
	fig = tools.make_subplots(rows = num_of_qubits, cols = num_of_parameters, subplot_titles = titles)
	fig['layout'].update(height=400*num_of_qubits, width=400*num_of_parameters)
	for index, qubit in enumerate(states.keys()):
		name_index = 1
		for parameter_id, key in enumerate(state.datasets.keys()):
			if 'Re' in titles[parameter_id]:
				yaxis=dict(domain = [index/num_of_qubits, (index + 0.7)/num_of_qubits])#, color='black'))
				xaxis=dict(domain = [parameter_id/num_of_parameters, (parameter_id+0.9)/num_of_parameters])#, color='black'))
				fig['layout']['xaxis'+str(parameter_id+1+index*num_of_parameters)].update(xaxis)
				fig['layout']['yaxis'+str(parameter_id+1+index*num_of_parameters)].update(yaxis)
				fig['layout']['annotations'][name_index]['x'] = (parameter_id+0.5)/num_of_parameters
				fig['layout']['annotations'][name_index]['y'] = (index + 0.75)/num_of_qubits
				name_index += 1
			if 'Im' in titles[parameter_id]:
				yaxis=dict(domain = [index/num_of_qubits, (index + 0.7)/num_of_qubits])#, color='black'))
				xaxis=dict(domain = [(parameter_id + 1)/num_of_parameters, (parameter_id+1.9)/num_of_parameters])#, color='black'))
				fig['layout']['xaxis'+str(parameter_id+1+index*num_of_parameters)].update(xaxis)
				fig['layout']['yaxis'+str(parameter_id+1+index*num_of_parameters)].update(yaxis)
				fig['layout']['annotations'][name_index]['x'] = (parameter_id+1.5)/num_of_parameters
				fig['layout']['annotations'][name_index]['y'] = (index + 0.75)/num_of_qubits
				name_index += 1
	for i in range(num_of_qubits):
		for j in range(num_of_parameters):
			fig.append_trace(trace[i][j], i+1, j+1)
	fig['layout']['xaxis'].update(title=state.datasets[key].parameters[1].name + ', ' + state.datasets[key].parameters[1].unit, titlefont=dict(size=14))
	fig['layout']['yaxis'].update(title=state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit, titlefont=dict(size=14))
	#fig['layout']
	print(fig)
	
	app.layout = html.Div(children=[
		html.H1(state.measurement_type, style={'fontSize': 30}),
		html.Div(html.P('Started at: ' + str(state.start)), style={'fontSize': 18}),
		html.Div(html.P('Stopped at: ' + str(state.stop)), style={'fontSize': 18}),
		html.Div(html.P('Owner: ' + str(state.owner)), style={'fontSize': 18}),
		html.Div(html.P('Metadata: ' + str(state.metadata)), style={'fontSize': 18}),
		#html.Div([a for a in state.metadata.items()])
		dcc.Graph(figure=fig)
	])
	app.run_server(debug=False)


def plot_with_dash_1D(state):
	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
	app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
	trace = []
	num_of_rows = len(state.datasets.keys())
	titles = []
	for key in state.datasets.keys():
		trace_help = []
		trace_help.append(go.Scatter(
			x = state.datasets[key].parameters[0].values,
            y = np.real(state.datasets[key].data),))
			#line = dict(color = 'blue')))
		trace_help.append(go.Scatter(
			x = state.datasets[key].parameters[0].values,
            y = np.imag(state.datasets[key].data),))
			#line = dict(color = 'blue')))
		titles.append('Re(' + str(key) + ')')
		titles.append('Im(' + str(key) + ')')
		trace.append(trace_help)
	fig = tools.make_subplots(rows = num_of_rows, cols = 2, subplot_titles = titles)
	for i in range(num_of_rows):
		for j in range(2):
			fig.append_trace(trace[i][j], i+1, j+1)
	yaxis=dict(title=key, titlefont=dict(size=16))#, color='black'))
	xaxis=dict(title=state.datasets[key].parameters[0].name + ', ' + state.datasets[key].parameters[0].unit, titlefont=dict(size=16))#, color='black'))
	for i in range (num_of_rows*2):
		fig['layout']['xaxis'+str(i+1)].update(xaxis)
		fig['layout']['yaxis'+str(i+1)].update(yaxis)

	app.layout = html.Div(children=[
		html.H1(state.measurement_type, style={'fontSize': 30}),
		html.Div(html.P('Started at: ' + str(state.start)), style={'fontSize': 18}),
		html.Div(html.P('Stopped at: ' + str(state.stop)), style={'fontSize': 18}),
		html.Div(html.P('Owner: ' + str(state.owner)), style={'fontSize': 18}),
		html.Div(html.P('Metadata: ' + str(state.metadata)), style={'fontSize': 18}),
		#html.Div([a for a in state.metadata.items()])
		dcc.Graph(figure=fig)
	])
	app.run_server(debug=False)