from mido import MidiFile, MidiTrack, Message
import mido
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.models import load_model
import os
from tqdm import *

import pygame
import IPython
import matplotlib.pyplot as plt
import librosa.display
from IPython import *
from music21 import *
from music21 import converter, instrument, note, chord, stream, midi
import glob
import time
import numpy as np
import keras.utils as utils
import pandas as pd
from music21 import *
from sklearn.model_selection import train_test_split
from tensorflow import distribute
import struct
import base64
import json
import site

strategy = distribute.MirroredStrategy(["GPU:0", "GPU:1"])


# visualise the tracks in midi file.
# mid = MidiFile('./content/musicnet_midis/Mozart/2313_qt15_1.mid')

# for i in mid.tracks[1]:
#     print(i)


us = environment.UserSettings()
us['musescoreDirectPNGPath'] = '/Users/Macbook_Autonomous/mscore'
us['directoryScratch'] = '/tmp'

# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.


def streamToNoteArray(stream):
    """
    Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
        0-127 - note on at specified pitch
        128   - note off
        129   - no event
    """
    # Part one, extract from stream
    total_length = np.int(
        np.round(stream.flat.highestTime / 0.25))  # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append([np.round(
                element.offset / 0.25), np.round(element.quarterLength / 0.25), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset / 0.25), np.round(
                element.quarterLength / 0.25), element.sortAscending().pitches[-1].midi])
    np_stream_list = np.array(stream_list, dtype=np.int)
    df = pd.DataFrame(
        {'pos': np_stream_list.T[0], 'dur': np_stream_list.T[1], 'pitch': np_stream_list.T[2]})
    # sort the dataframe properly
    df = df.sort_values(['pos', 'pitch'], ascending=[True, False])
    df = df.drop_duplicates(subset=['pos'])  # drop duplicate values
    # part 2, convert into a sequence of note events
    # set array full of no events by default.
    output = np.zeros(total_length+1, dtype=np.int16) + \
        np.int16(MELODY_NO_EVENT)
    # Fill in the output list
    for i in range(total_length):
        if not df[df.pos == i].empty:
            # pick the highest pitch at each semiquaver
            n = df[df.pos == i].iloc[0]
            output[i] = n.pitch  # set note on
            output[i+n.dur] = MELODY_NOTE_OFF
    return output


def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df['offset'] = df.index
    df['duration'] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    # calculate durations and change to quarter note fractions
    df.duration = df.duration.diff(-1) * -1 * 0.25
    df = df.fillna(0.25)
    return df[['code', 'duration']]


def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            # bit of an oversimplification, doesn't produce long notes.
            new_note = note.Rest()
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream


# wm_mid = converter.parse("./content/musicnet_midis/Mozart/2313_qt15_1.mid")
# wm_mid.show()
# wm_mel_rnn = streamToNoteArray(wm_mid)
# noteArrayToStream(wm_mel_rnn).show()


# gettinng the note on values from the messages on 50 midi files
note_on = []
n = 50
filenames = os.listdir('./content/musicnet_midis/Mozart')
for filename in filenames:
    mid = MidiFile('./content/musicnet_midis/Mozart/' + filename)
    for j in range(len(mid.tracks)):
        for i in mid.tracks[j]:
            if str(type(i)) != "<class 'mido.midifiles.meta.MetaMessage'>":
                x = str(i).split(' ')
                if x[0] == 'note_on':
                    note_on.append(int(x[2].split('=')[1]))

inputlen = 20
# making data to train
training_data = []
labels = []
for i in range(inputlen, len(note_on)):
    training_data.append(note_on[i-inputlen:i])
    labels.append(note_on[i])


different_labels = set(labels)


model = Sequential()

model.add(LSTM(64, input_shape=(inputlen, 1), unroll=True,
          return_sequences=True, implementation=1))
model.add(Dropout(0.4))
# model.add(LSTM(64))
# model.add(Dense(64, 'relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, 'relu'))

model.compile(loss='MSE', optimizer='adam')
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss', patience=20, verbose=0)

training_data = np.array(training_data)
training_data = training_data.reshape(
    (training_data.shape[0], training_data.shape[1], 1))
labels = np.array(labels)


# train
X_train, X_test, y_train, y_test = train_test_split(
    training_data, labels, test_size=0.05, random_state=42)
model.fit(X_train, y_train, epochs=200, batch_size=32 * strategy.num_replicas_in_sync,
          validation_data=(X_test, y_test), callbacks=[early_stop])
model.save('musicnetgen.h5')

# or load trained model
# model = load_model('musicnetgen.h5')

# prediction

n = 10
# randomize the starter notes
index = np.random.choice(len(training_data))
starter_notes = training_data[index]
x = starter_notes.reshape(1, inputlen, 1)
# print("starter notes: ", starter_notes)
tune = list(starter_notes.reshape(-1,))
for i in range(n):
    out = model.predict(x)
    # print("output shape: ", out.shape)
    pred = int(out[0][0])
    if round(pred) == round(tune[-1]):
        p = np.random.choice(['a', 'b', 'c'])
        if p == 'a':
            pred = 65
        elif p == 'b':
            pred = 60
        else:
            pred = 70
    tune.append(pred)
    x = tune[-inputlen:]
    x = np.array(x)
    x = x.reshape(1, inputlen, 1)

tune = list(np.array(tune).astype('float32'))

# encoder

offset = 0

output_notes = []
output_melody_stream = stream.Stream()
# create note and chord objects based on the values generated by the model
for patterns in tune:
    pattern = str(patterns)
    # pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
        output_melody_stream.append(new_chord)
    # pattern is a note
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        output_melody_stream.append(new_note)
    # increase offset each iteration so that notes do not stack
    offset += 0.5

# write to midi file
output_melody_stream.write('midi', fp='test_output.mid')
# print(output_melody_stream)

# print(output_melody_stream)
# output_melody_stream.show()


def get_weights(model):
    weights = []
    for layer in model.layers:
        w = layer.get_weights()
        print(layer.name)
        print([g.shape for g in w])
        weights.append(w)
    return weights


def compressConfig(data):
    layers = []
    for layer in data["config"]["layers"]:
        cfg = layer["config"]
        if layer["class_name"] == "InputLayer":
            layer_config = {
                "batch_input_shape": cfg["batch_input_shape"]
            }
        elif layer["class_name"] == "Rescaling":
            layer_config = {
                "scale": cfg["scale"],
                "offset": cfg["offset"]
            }
        elif layer["class_name"] == "Dense":
            layer_config = {
                "units": cfg["units"],
                "activation": cfg["activation"]
            }
        elif layer["class_name"] == "Conv2D":
            layer_config = {
                "filters": cfg["filters"],
                "kernel_size": cfg["kernel_size"],
                "strides": cfg["strides"],
                "activation": cfg["activation"],
                "padding": cfg["padding"]
            }
        elif layer["class_name"] == "MaxPooling2D":
            layer_config = {
                "pool_size": cfg["pool_size"],
                "strides": cfg["strides"],
                "padding": cfg["padding"]
            }
        elif layer["class_name"] == "Embedding":
            layer_config = {
                "input_dim": cfg["input_dim"],
                "output_dim": cfg["output_dim"]
            }
        elif layer["class_name"] == "SimpleRNN":
            layer_config = {
                "units": cfg["units"],
                "activation": cfg["activation"]
            }
        elif layer["class_name"] == "LSTM":
            layer_config = {
                "units": cfg["units"],
                "activation": cfg["activation"],
                "recurrent_activation": cfg["recurrent_activation"],
            }
        else:
            layer_config = None

        res_layer = {
            "class_name": layer["class_name"],
        }
        if layer_config is not None:
            res_layer["config"] = layer_config
        layers.append(res_layer)

    return {
        "config": {
            "layers": layers
        }
    }


# TODO: parameterize
def get_model_for_export(model):
    weight_np = get_weights(model)

    weight_bytes = bytearray()
    for idx, layer in enumerate(weight_np):
        # print(layer.shape)
        # write_to_file(f"model_weight_{idx:02}.txt", str(layer))
        for weight_group in layer:
            flatten = weight_group.reshape(-1).tolist()
            # print("flattened length: ", len(flatten))
            for i in flatten:
                weight_bytes.extend(struct.pack("@f", float(i)))

    weight_base64 = base64.b64encode(weight_bytes).decode()
    config = json.loads(model.to_json())
    # print("full config: ", config)

    compressed_config = compressConfig(config)
    # write to file
    with open("model_config.json", "w") as f:
        json.dump({
            "model_name": "musicnetgen",
            "layers_config": compressed_config,
            "weight_b64": weight_base64,
        }, f)
    return weight_base64, compressed_config


get_model_for_export(model)
