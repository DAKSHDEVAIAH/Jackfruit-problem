import os
from bing_image_downloader import downloader
import shutil

# Configuration
DATASET_DIR = 'datasets'
SAMPLES_PER_CLASS = 20
FLOWER_TYPES = [
    'Rose', 'Sunflower', 'Tulip', 'Daisy', 
    'Lavender', 'Marigold', 'Violet', 'Lily'
]

def scrape_images():
    print("Starting image scraping with Bing...")
    
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    for flower in FLOWER_TYPES:
        print(f"Processing {flower}...")
        query = f"{flower} flower"
        
        # Output dir for this flower
        # bing_image_downloader creates a subfolder with the query name
        # We want it in datasets/FlowerName
        
        # It downloads to output_dir/query_string
        # So we set output_dir to datasets/temp and move them?
        # Or just let it download to datasets/FlowerName?
        
        # Let's try to download to datasets/ directly, but the folder name will be the query string.
        # So we will download to a temp folder and move/rename.
        
        try:
            downloader.download(
                query, 
                limit=SAMPLES_PER_CLASS, 
                output_dir=DATASET_DIR, 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=60, 
                verbose=False
            )
            
            # The downloader creates a folder 'datasets/FlowerName flower'
            # We want 'datasets/FlowerName'
            
            downloaded_dir = os.path.join(DATASET_DIR, query)
            target_dir = os.path.join(DATASET_DIR, flower)
            
            if os.path.exists(downloaded_dir):
                if os.path.exists(target_dir):
                    # Move files from downloaded_dir to target_dir
                    for file in os.listdir(downloaded_dir):
                        shutil.move(os.path.join(downloaded_dir, file), os.path.join(target_dir, file))
                    os.rmdir(downloaded_dir)
                else:
                    os.rename(downloaded_dir, target_dir)
            
            print(f"  Finished {flower}")
            
        except Exception as e:
            print(f"  Error searching for {flower}: {e}")

if __name__ == "__main__":
    scrape_images()
