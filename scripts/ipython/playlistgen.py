import numpy as np
import os 
import pickle

class Song:

	def __init__(self, name, artiste, feat):
		self.title = name
		self.artist = artiste
		self.features = feat

	def getTitle(self):
		return self.title

	def getArtist(self):
		return self.artist

	def getFeature(self, name):
		return self.features[name]

	def addFeature(self, dictl):
		self.features = dictl

class PlaylistGen:
	def __init__(self, distance, i, l, length, layer0):
		self.dtype = distance
		self.iterations = i 
		self.lambd = l 
		self.plength = length
		self.layer0factor = layer0
		self.u = [] 
		self.r = []
		self.k = 0

	def suggest(self, name, author):
		firstlayer = SongSuggestion(self.dtype, self.plength*self.layer0factor)
		layer0 = firstlayer.suggest(name, author)