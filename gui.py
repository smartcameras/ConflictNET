import math
import PySimpleGUI as sg
import numpy as np
import sounddevice as sd
import librosa

def find_closest(p,vals):
	x = p[0]
	y = p[1]
	X = np.array(vals[0])
	Y = np.array(vals[1])
	l = np.sqrt(np.square(X - x) + np.square(Y - y))
	if(l.min()<=4):
		idx = np.where(l == l.min())
		s_id = idx[0]
		x = X[s_id]
		y = Y[s_id]
	else:
		s_id = []
		x = []
		y = []
	return s_id,x,y

layout = [[sg.Graph(canvas_size=(800, 800), graph_bottom_left=(-105,-105), graph_top_right=(105,105), background_color='white', key='graph', tooltip=None, enable_events=True)],]
window = sg.Window('ConflictNET Prediction Evaluation', layout, grab_anywhere=True).Finalize()
graph = window['graph']
# Draw axis
graph.DrawLine((-100,0), (100,0))
graph.DrawLine((0,-100), (0,100))
for x in range(-100, 101, 20):
	graph.DrawLine((x,-3), (x,3))
	if x != 0:
		graph.DrawText(x, (x,-10), color='green')
for y in range(-100, 101, 20):
	graph.DrawLine((-3,y), (3,y))
	if y != 0:
		graph.DrawText(y, (-10,y), color='blue')
# Draw Graph
f= open("./data/train_analysis.txt","r")
x_tr = np.load('./data/x_train.npy')
#s_id = []
true_val = []
pred_val = []
f.readline() # to remove column names
for x in f:
	s = x.split()
	#s_id.append(float(s[0]))
	true_val.append(int(round(float(s[1])*100)))
	pred_val.append(int(round(float(s[2])*100)))
#i=100
vals = [true_val,pred_val]
for i in range(len(true_val)):
	#graph.DrawLine((true_val[i],0),(true_val[i],pred_val[i]))
	graph.DrawCircle((true_val[i],pred_val[i]),4,line_color='red', fill_color='blue')
while True:
	event, values = window.read()
	if event is None:
		break
	#print(event, values)
	val = values[event]
	print(val)
	# find closest data point 
	s_id, x, y = find_closest(val,vals)
	if (len(s_id) != 0):
		print('Sample ID:',s_id)
		sig = x_tr[s_id]
		sig = np.reshape(sig,(240000))
		#Getting an error using 8k with sd [Error: PaAlsaStreamComponent_BeginPolling: Assertion `ret == self->nfds' failed. Aborted (core dumped)]. So using 16k for playback
		aud = librosa.resample(sig,8000,16000)
		sd.play(aud,16000,blocking=True)
