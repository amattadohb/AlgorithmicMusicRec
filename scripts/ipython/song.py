

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

