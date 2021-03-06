from playlistgen import *
from playlistgeneration import *

# generator = PlaylistGen('euc', 10, 0.0, 15, 3)
# songs = generator.suggest('Meme Generator', 'Dan Deacon ', 'sr', 'zcr', '', '')

distances = ['euc', 'max']
iterations = 10
lambduh = 0.0
playlist_length = 15
playlist_length_factor = 3
songs = [('Meme Generator', 'Dan Deacon ')]
features = ['sr', 'sc', 'rms', 'zcr', 'mfcc']
'''
for song in songs:
	for distance in distances:
		for feat1 in features:
			for feat2 in features:
				if feat2 == feat1:
					continue
				for feat3 in features:
					if feat3 == feat1 or feat3 == feat2:
						continue
					for feat4 in features:
						if feat4 == feat1 or feat4 == feat2 or feat4 == feat3:
							continue
						print 'Playlist for\t', song[1], song[0], feat1, feat2, feat3, feat4
						generator = PlaylistGen(distance, iterations, lambduh, playlist_length, playlist_length_factor)
						generator.suggest(song[0], song[1], feat1, feat2, feat3, feat4)
'''

features = ['sr', 'rms', 'zcr']
distance = 'euc'
ranges = [0, 3, 7]
for feat3 in features:
	for feat4 in features:
		for r in ranges:
			if feat4 == feat3:
				continue
			if feat3 == 'rms' and feat4 == 'zcr' or feat4 == 'rms' and feat3 == 'zcr':
				continue
			print feat3, feat4, r
			generator = PlaylistGen(distance, iterations, lambduh, playlist_length, playlist_length_factor)
			generator.suggest('Monster', 'Kanye West, Jay-Z, Rick Ross, Nicki Minaj & Bon Iver', 'mfcc', 'sc', feat3, feat4, r)
			print '\n\n\n'

#generator = PlaylistGen('euc', 10, 0.0, 10, 3)
#generator.suggest('Meme Generator', "Dan Deacon ", "rms", "sr", "zcr", "mfcc")
