from playlistgeneration import *

generator = PlaylistGeneration('euc', 10, 0.0)
generator.train('sr', 'sc')
generator.predict('Broken Jaw', 'Foster the People', 0, 50, 'sr', 'sc')