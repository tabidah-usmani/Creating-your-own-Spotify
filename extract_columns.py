import pandas as pd
import subprocess
import io

def add_leading_zeros(track_id):
    # Convert track_id to string and add leading zeros
    return str(track_id).zfill(6)

def main():
    # Use subprocess to run the cat command to read the file contents
    process = subprocess.Popen(["cat", "/home/mary/Downloads/raw_tracks.csv"], stdout=subprocess.PIPE)
    output, _ = process.communicate()

    # Check if output is empty
    if output:
        # Read CSV from the command output, specifying only the desired columns
        columns_to_read = ['track_id', 'track_title', 'album_title', 'artist_name']
        df = pd.read_csv(io.BytesIO(output), usecols=columns_to_read, encoding='utf-8')

        # Rename columns to match the desired output
        df.columns = ['track_id', 'Title', 'Album', 'Contributing artists']

        # Apply the function to add leading zeros to the track_id column
        df['track_id'] = df['track_id'].apply(add_leading_zeros)

        # Print first few rows of the DataFrame
        print(df)
    else:
        print("No data found in the file.")

if __name__ == "__main__":
    main()

