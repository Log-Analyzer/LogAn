import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.telemetry'))

def get_elasticsearch_config():
    return {
        'host': os.getenv('ELASTICSEARCH_HOST', 'localhost'),
        'port': os.getenv('ELASTICSEARCH_PORT', 443),
        'username': os.getenv('ELASTICSEARCH_USERNAME'),
        'password': os.getenv('ELASTICSEARCH_PASSWORD'),
    }

def get_environment():
    return os.getenv('ENVIRONMENT', 'LOCAL')

def get_stats_index():
    env = get_environment()
    index = os.getenv(f'ES_STATS_{env}_INDEX', 'stats-local')
    return index


def get_feedback_index():
    env = get_environment()
    index = os.getenv(f'ES_FEEDBACK_{env}_INDEX', 'feedback-local')
    return index

def get_es_client():
    config = get_elasticsearch_config()

    es = Elasticsearch(
        config['host'],
        basic_auth=(config['username'], config['password']),
        verify_certs=False
    )

    #  Authenticate via the .options() method:
    ecs = es.options(
        basic_auth=(config['username'], config['password'])
    )

    return ecs


def create_document(index, document):
    es = get_es_client()
    response = es.index(index=index, document=document)
    return response