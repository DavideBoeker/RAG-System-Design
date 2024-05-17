# Import Libraries
from chromadb import HttpClient

client = HttpClient(host="localhost", port=8000)

print('HEARTBEAT:', client.heartbeat())