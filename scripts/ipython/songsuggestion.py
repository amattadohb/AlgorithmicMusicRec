import pickle
import os
from heapq import *
from scipy.spatial import distance

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

class SongSuggestion:

	def __init__(self, dist, num_song):
		self.dtype = dist
		self.num_songs = num_song
		self.song_heap = []

		self.library = []
		#add all songs to the library
		for subpath, dirs, filelist in os.walk(os.path.join('./music', 'pickle')):
			for filename in filelist:
				if filename[-4:] == '.p':
					toAdd = pickle.load(open(os.path.join(os.path.join(subpath, filename) , "rb" )))
					self.library.append(toAdd)


	def suggest(self, name, author): 
		seed_title = name
		seed_artist = author
		
		#get song object of seed song
		seed = pickle.load(open(os.path.join('music', 'pickle', seed_artist + '_' + seed_title + '.p') , "rb" ))

		for song in self.library:
			#don't include the seed song or other songs by the same artist
			if song.getArtist() != seed_artist:
				dist1 = getDist(seed, song, 'sc')
				dist2 = getDist(seed, song, 'sr')
				dist = (dist1 + dist2 / 2.0)

			heappush(song_heap, (dist, song))

		suggestions = nsmallest(self.num_songs, song_heap)
		print("Suggested songs for " + seed_title + " by " + seed_artist)
		for s in suggestions:
			print(s.getTitle() + s.getArtist())



	def getDist(self, seed, compare, feat):
		seed_feat = seed.getFeature(feat)
		compare_feat = compare.getFeature(feat)

		#using euclidean distance
		if self.dtype == 'euc':
			return distance.euclidean(seed_feat, compare_feat)

		#using maximum distance
		if self.dtype == 'max':
			max_dist = 0

			#find max distance
			for index in range(0, len(seed_feat)):
				dist = abs(seed_feat[index] - compare_feat[index])
				if dist > max_dist:
					max_dist = dist

			return max_dist

