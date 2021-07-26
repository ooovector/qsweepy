from qsweepy.ponyfiles import save_exdir
import numpy as np
#import webcolors
from pony.orm import *
import plotly.io as pio
import pandas as pd

def save_default_plot(state, db):
	plot = default_plot(state, db)
	pio.write_image(plot, state.filename+'.svg', width=1200, height=900)
	pio.write_image(plot, state.filename+'.png', width=1200, height=900)

def default_plot(state, db):
	measurements = [{'id': state.id}]
	measurements = add_fits_to_measurements(measurements, db)
	traces, dropdowns = add_default_traces(measurements, db, interactive=False)
	traces = pd.DataFrame(traces, columns=['id', 'dataset', 'op', 'style', 'color', 'x-axis', 'y-axis', 'row', 'col'])
	cross_sections = cross_section_configurations_add_default(traces, db)
	return plot(traces, cross_sections, db)

def cross_section_configurations_add_default(selected_traces, db, current_config=[]):
	measurements_to_load = selected_traces['id'].unique()
	measurements = {}
	measurement_types = []
	cross_sections = []
	# load measurements
	with db_session:
		for measurement_id in measurements_to_load:
			measurements[measurement_id] = save_exdir.load_exdir(db.Data[int(measurement_id)].filename, db, lazy=True)
			measurement_types.append(measurements[measurement_id].measurement_type)

	for trace_id, trace in selected_traces.to_dict('index').items():
		dataset_name = trace['dataset']
		measurement = measurements[trace['id']]
		dataset = measurement.datasets[dataset_name]
		for parameter_id, parameter in enumerate(dataset.parameters):
			if parameter.name != trace['x-axis'] and parameter.name != trace['y-axis']:
				current_config_cell = [c for c in current_config if c['trace-id'] == trace_id and c['parameter-id'] == parameter_id]
				if len(current_config_cell):
					current_value = current_config_cell[0]['value']
				else:
					current_value = parameter.values[0]
				cross_sections.append({'trace-id':trace_id, 'parameter-id':parameter_id, 'parameter':parameter.name, 'value':0})
	for measurement_id, measurement in measurements.items():
		measurement.exdir.close()
	return cross_sections

def add_default_traces(loaded_measurements, db, old_traces=[], conditional_dropdowns=[], interactive=True):
	'''
	Generate conditional dropdown filter queries for x-axis and y-axis

	:param loaded_measurements:
	:param db:
	:param old_traces:
	:param conditional_dropdowns: list of conditional dropdown filter queries
	:return: data and list of conditional dropdowns for x-axis and y-axis
	'''
	data = old_traces #[i for i in old_traces]
	measurement_signatures_list = list(set([(d['dataset'], d['x-axis'], d['y-axis'], d['op']) for d in old_traces]))
	measurement_signatures = {measurement_signature:[d for d in old_traces if (d['dataset'], d['x-axis'], d['y-axis'], d['op']) == measurement_signature] for measurement_signature in measurement_signatures_list}
	measurement_signatures_axes_mapping = {measurement_signature:{'row':data[-1]['row'], 'col':data[-1]['col']} for measurement_signature, data in measurement_signatures.items()}
	measurement_signatures_new = {}
	cond_column = "{dataset}"

	with db_session:
		for m in loaded_measurements:
			measurement_id = m['id']
			measurement_state = save_exdir.load_exdir(db.Data[int(measurement_id)].filename, db, lazy=True)
			for dataset in measurement_state.datasets.keys():
				if len(measurement_state.datasets[dataset].parameters) < 1:
					continue
				parameter_names = [p.name for p in measurement_state.datasets[dataset].parameters]
				dropdown_row_condition = f'{cond_column} eq "{dataset}"'
				if np.iscomplexobj(measurement_state.datasets[dataset].data): # if we are dealing with complex object, give the chance of selecting which op we want to apply
					operations = ['Re', 'Im', 'Abs', 'Ph']
				else:
					operations = ['']
				# check if there is a condition already
				if not (len([True for d in conditional_dropdowns if d['if']['filter_query']==dropdown_row_condition])):
					# construct filter query for conditional dropdowm
					for axis in ['x-axis', 'y-axis']:
						conditional_dropdowns.append({'if': {'column_id': axis, 'filter_query': dropdown_row_condition},
												  'options': [{'label': p, 'value': p} for p in parameter_names] + [
													  {'label': 'data', 'value': 'data'}]})

				#y_axis_conditional_dropdowns.append({'condition':y_dropdown_row_condition, 'dropdown':[{'label': p, 'value': p} for p in parameter_names]+[{'label':'data', 'value':'data'}]})
				if len(measurement_state.datasets[dataset].data.shape) > 0:
					x_default = measurement_state.datasets[dataset].parameters[np.argsort(measurement_state.datasets[dataset].data.shape)[-1]].name
				else:
					x_default = 'data'

				y_default = 'data'
				style_default = '-'
				if len(measurement_state.datasets[dataset].data.shape) > 1:
					if measurement_state.datasets[dataset].data.shape[np.argsort(measurement_state.datasets[dataset].data.shape)[-2]]>1:
						y_default = measurement_state.datasets[dataset].parameters[np.argsort(measurement_state.datasets[dataset].data.shape)[-2]].name
						style_default = '2d';
				for operation in operations:
					measurement_signature = (dataset, x_default, y_default, operation)
					if measurement_signature in measurement_signatures_axes_mapping:
						row = measurement_signatures_axes_mapping[measurement_signature]['row']
						col = measurement_signatures_axes_mapping[measurement_signature]['col']
					else:
						row = 0
						col = 0
						if measurement_signature not in measurement_signatures_new:
							measurement_signatures_new[measurement_signature] = [len(data)]
						else:
							measurement_signatures_new[measurement_signature].append(len(data))
					row = {'id': measurement_id,
					   'dataset': dataset,
					   'op':operation,
					   'style': style_default,
					   'color': 'salmon',
					   'x-axis': x_default,
					   'y-axis': y_default, #[1]?
					   'row': row,
					   'col': col}
					data.append(row)
			measurement_state.exdir.close()

	for measurement_signature_id, data_ids in enumerate(measurement_signatures_new.values()):
		for data_id in data_ids:
			data[data_id]['row'], data[data_id]['col'] = ax_id(measurement_signature_id+len(measurement_signatures), len(measurement_signatures_new)+len(measurement_signatures))
				# check if dataset is in current traces, if not, update the cell values with current values
	return data, conditional_dropdowns

def default_measurements(db):
	with db_session:
		last_measurement = select((measurement.id, measurement.filename) for measurement in db.Data).order_by(lambda id,filename: desc(id)).first()
		last_meaningful_measurement = last_measurement
		if last_meaningful_measurement is None:
			return []
		while (len(save_exdir.load_exdir(last_meaningful_measurement[1], db).datasets) == 0):
			last_meaningful_measurement = select((measurement.id, measurement.filename) for measurement in db.Data if measurement.id < last_meaningful_measurement[0]).order_by(lambda id,filename: desc(id)).first()

		if last_measurement != last_meaningful_measurement:
			data = [{'id': last_meaningful_measurement[0], 'label': 'current'}, {'id': last_measurement[0], 'label': 'current_meta'}]
		else:
			data = [{'id': last_meaningful_measurement[0], 'label': 'current'}]
	return add_fits_to_measurements(data, db)

def add_fits_to_measurements(measurements, db):
	extra_measurements = []
	with db_session:
		for measurement in measurements:
			references = db.Data[measurement['id']].reference_one
			for r in references:
				if r.ref_type == 'fit source':
					extra_measurements.append({'id': r.that.id, 'label':measurement['label']+' fit source'})
			references = db.Data[measurement['id']].reference_two
			for r in references:
				if r.ref_type == 'fit source':
					extra_measurements.append({'id': r.this.id, 'label':measurement['label']+' fit'})
	return measurements+extra_measurements


def ax_id(ax_id, ax_num):
	if ax_num ==1:
		return (0,0)
	elif ax_num <= 3:
		return (0, ax_id)
	elif ax_num <= 8:
		return (ax_id%2, ax_id//2)
	elif ax_num <= 9:
		return (ax_id//3, ax_id%3)
	elif ax_num <= 12:
		return (ax_id//4, ax_id%4)
	else:
		cols = int(np.ceil(np.sqrt(ax_num)))
		return (ax_id//cols, ax_id%cols)

def plot(selected_traces, cross_sections, db):
	from time import time
	start_time = time()
	measurements_to_load = selected_traces['id'].unique()
	measurements = {}
	measurement_types = []
	# load measurements
	with db_session:
		for measurement_id in measurements_to_load:
			measurements[measurement_id] = save_exdir.load_exdir(db.Data[int(measurement_id)].filename, db, lazy=True)
			measurement_types.append(measurements[measurement_id].measurement_type)

	load_time = time()
	print ('load time: ', load_time - start_time)
	if len(selected_traces['row']):
	# building subplot grid
		num_rows = int(selected_traces['row'].astype(int).max()+1)
	else:
		num_rows = 1

	if len(selected_traces['col']):
		num_cols = int(selected_traces['col'].astype(int).max()+1)
	else:
		num_cols = 1

	layout = {}
	figure = {}
	layout['annotations'] = []
	layout['showlegend'] = False
	layout['title'] = ', '.join(measurement_types)
	figure['data'] = []

	if num_cols < 3:
		x_offset = 0.1
	elif num_cols < 5:
		x_offset = 0.13
	elif num_cols < 7:
		x_offset = 0.15
	else:
		x_offset = 0.2

	if num_rows < 3:
		y_offset = 0.1
	else:
		y_offset = 0.2

	for row in range(num_rows):
		for col in range(num_cols):
			layout['xaxis{}'.format(row*num_cols+col+1)] = {'anchor': 'y{}'.format(row*num_cols+col+1),
															'domain': [(col+x_offset)/num_cols, (col + 1.0 - x_offset)/num_cols],}
			layout['yaxis{}'.format(row*num_cols+col+1)] = {'anchor': 'x{}'.format(row*num_cols+col+1),
															'domain': [(row+y_offset)/num_rows, (row + 1.0 - y_offset)/num_rows], }

	pre_trace_time = time()
	print ('pre_trace_time: ', pre_trace_time - load_time)

	for trace_id, trace in selected_traces.to_dict('index').items():
		trace_start = time()
		dataset_name = trace['dataset']
		measurement = measurements[trace['id']]
		dataset = measurement.datasets[dataset_name]
		x_axis_id = -1
		y_axis_id = -1
		title_x = dataset_name
		title_y = dataset_name

		indexes = [slice(None, None, None)]*len(dataset.data.shape)
		for parameter_values in cross_sections:
			if parameter_values['trace-id'] == trace_id:
				diff = np.asarray(dataset.parameters[parameter_values['parameter-id']].values)-float(parameter_values['value'])
				cross_section_id = np.argmin(np.abs(diff))
				indexes[parameter_values['parameter-id']] = cross_section_id

		for parameter_id, parameter in enumerate(dataset.parameters):
			if parameter.name == trace['x-axis']:
				dataset_x = np.memmap.tolist(parameter.values)
				title_x = '{name} ({unit})'.format(name=parameter.name, unit=parameter.unit)
				x_axis_id = parameter_id
			if parameter.name == trace['y-axis']:
				dataset_y = np.memmap.tolist(parameter.values)
				title_y = '{name} ({unit})'.format(name=parameter.name, unit=parameter.unit)
				y_axis_id = parameter_id

		data_flat = dataset.data[indexes]
		if trace['style'] != '2d':
			trace_data = data_flat
		else:
			if x_axis_id>y_axis_id:
				trace_data = data_flat.T
			else:
				trace_data = data_flat

		if trace['op'] == 'Im': data_to_plot = np.imag(trace_data)
		elif trace['op'] == 'Re': data_to_plot = np.real(trace_data)
		elif trace['op'] == 'Abs': data_to_plot = np.abs(trace_data)
		elif trace['op'] == 'Ph': data_to_plot = np.angle(trace_data)
		else: data_to_plot = trace_data

		x = dataset_x if x_axis_id != -1 else data_to_plot
		y = dataset_y if y_axis_id != -1 else data_to_plot
		#print ('new trace shape:', trace_data.shape, 'x shape:',np.asarray(x).shape, 'y shape:', np.asarray(y).shape)


		row = int(trace['row'])
		col = int(trace['col'])
		if trace['style'] == '-': style = 'lines'
		elif trace['style'] == 'o': style = 'markers'
		elif trace['style'] == '.': style = 'markers'
		else: style = '2d'

		plot_trace = {'type': 'heatmap' if style == '2d' else 'scatter',

					  #'mode': style,
					  #'marker': {'size': 5 if trace['style'] == 'o' else 2, 'color':trace['color']},
					  #'color': 'rgb({red},{blue},{green})'.format(red=webcolors.name_to_rgb(trace['color']).red,
					  #											  blue=webcolors.name_to_rgb(trace['color']).blue,
					  #											  green=webcolors.name_to_rgb(trace['color']).green),
					  'xaxis': 'x{}'.format(row*num_cols+col+1),
					  'yaxis': 'y{}'.format(row*num_cols+col+1),
					  'x': x,
					  'y': y}
		if style == '2d':
			plot_trace['colorbar'] = {'len': (1.0-y_offset*2)/num_rows,
							   'thickness': 0.025/num_cols,
							   'thicknessmode': 'fraction',
							   'x': (col + 1.0-x_offset)/num_cols,
							   'y': (row + 0.5)/num_rows}
			plot_trace['colorscale'] = 'Blackbody'
			plot_trace['z'] = data_to_plot.T
		else:
			plot_trace['mode'] = style
			plot_trace['marker'] = {'size': 5 if trace['style'] == 'o' else 2, 'color':trace['color']}

		figure['data'].append(plot_trace)
		#print (figure['data'][-1]['xaxis'], figure['data'][-1]['yaxis'])

		layout['annotations'].append({'font': {'size': 16},
						'showarrow': False,
						'text': str(trace['id']) + ': ' + trace['op'] + '(' + dataset_name + ')',
						'x': (col + 0.5)/num_cols,
						'xanchor': 'center',
						'xref': 'paper',
						'y': (row + 1.0 - y_offset)/num_rows,
						'yanchor': 'bottom', 'yref': 'paper'})

		layout['xaxis{}'.format(row*num_cols+col+1)].update({'title': title_x})
		layout['yaxis{}'.format(row*num_cols+col+1)].update({'title': title_y})
		trace_end_time = time()
		print ('trace {} time: '.format(trace_id), trace_end_time - pre_trace_time)

		#print(layout['xaxis{}'.format(row*num_cols+col+1)], layout['yaxis{}'.format(row*num_cols+col+1)])
	figure['layout'] = layout

	for measurement_id, measurement in measurements.items():
		measurement.exdir.close()

	return figure
