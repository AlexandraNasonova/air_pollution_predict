import boto3
from settings import env_reader

if __name__ == "__main__":
    env_values = env_reader.get_params()
    s = env_values.get("S3_ACCESS_KEY_ID")
    b = env_values.get("S3_BUCKET")[5:]

    s3 = boto3.resource('s3',
                        aws_access_key_id=env_values.get("S3_ACCESS_KEY_ID"),
                        aws_secret_access_key=env_values.get("S3_SECRET_ACCESS_KEY"),
                        endpoint_url=env_values.get("S3_URI"))
    s3client = boto3.client('s3')
    bucket = s3.Bucket(env_values.get("S3_BUCKET")[5:])

    for obj in bucket.objects.all():
        #s3client.download_file(bucket_name, obj.key, filename)
        print('{0}:{1}'.format(bucket.name, obj.key))
