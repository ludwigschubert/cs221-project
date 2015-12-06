import random
import psycopg2

class DB(object):

	def getReadmes(self, num):
		"""
		Make connection to postgres server and get num readme entries
		We want to return a list of lists [[id, readme_text, stars, ...], [id, readme_text, stars, ...], ...]
		"""
		try:
			conn=psycopg2.connect("dbname='data-collector_development' user='friedemann'") # user='dbuser' password='mypass'")
		except:
			print "I am unable to connect to the database."

		cur = conn.cursor()
		try:
			#TODO: do we want a specific subset? Will this be repeatable?
			cur.execute('SELECT id, readme_html, stargazers_count FROM repositories WHERE repositories.readme_html IS NOT NULL AND repositories.stargazers_count IS NOT NULL LIMIT {}'.format(num))
		except:
			print "I can't SELECT from repositories"

		rows = cur.fetchall()

		# if (DEBUG_VERBOSITY > 2):
		# 	print "\nRows: \n"
		# 	for row in rows:
		# 		print "	", row[1]

		return rows # should be in correct form

	def loadRandomSamples(self, ntrain, ntest):
		'''
		Get a bunch of data from db.
		Randomly pick from your data (to simulate random draw)
		Assign ntrain to training set, ntest to testing set
		'''

		training_ids = set()
		testing_ids	 = set()

		training_data = []
		testing_data  = []

		data = self.getReadmes((ntrain+ntest)*10) # pick randomly from a pool 10x the total desired size

		while (len(training_data) < ntrain):
			testidx = random.randint(0,len(data)-1)
			if (testidx in training_ids or testidx in testing_ids):
				continue
			training_data.append(data[testidx])
			training_ids.add(testidx)

		while (len(testing_data) < ntest):
			testidx = random.randint(0,len(data)-1)
			if (testidx in training_ids or testidx in testing_ids):
				continue
			testing_data.append(data[testidx])
			testing_ids.add(testidx)
		return (training_data, testing_data)