import dropbox
from pathlib import Path
import os
from tqdm import tqdm

def download_data():
    """
    Downloads contents from specific Dropbox shared folders to ./data directory.
    Shows progress as percentage for each folder.
    """
    DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
    LOCAL_PATH = Path("./data")
    
    FOLDER_URLS = {
        "train": "https://www.dropbox.com/scl/fo/d92u3tl2swvykye6dzrnb/ANkbqHfRh2952CYWicBJD88?rlkey=ccevttml9h3o5s2annd19pc8u&st=3fhetp17",
        "test": "https://www.dropbox.com/scl/fo/uoyzjmoaxhblk97dtbag5/AP34U9sprG7bDaxZ54OhveM?rlkey=25w4aw1fpexrnmo1fmdk4tzg9&st=vlmiklib",
        "eval": "https://www.dropbox.com/scl/fo/oz919x1vnyxshna9sr879/ADMFoF423Gvas9ECNKRSGBI?rlkey=pvz2pm1rzvtnzzyxzv2hhy9bq&st=h48qgdty",
        "train_small": "https://www.dropbox.com/scl/fo/3yn6c9z7gp7eegq2r9kkp/AMTJBawt9RkJSGnGzrq4qxI?rlkey=an7qnnluyikazwy2ojn8dzh2i&st=oeqp0l7y"
    }

    def get_all_files(dbx, shared_link):
        """Get complete list of files handling pagination"""
        files = []
        result = dbx.files_list_folder(path="", shared_link=shared_link)
        
        # Get first batch of files
        files.extend([
            entry for entry in result.entries 
            if isinstance(entry, dropbox.files.FileMetadata)
        ])
        
        # Keep getting more files while available
        while result.has_more:
            result = dbx.files_list_folder_continue(result.cursor)
            files.extend([
                entry for entry in result.entries 
                if isinstance(entry, dropbox.files.FileMetadata)
            ])
            
        return files

    def download_folder_contents(dbx, folder_name: str, shared_url: str):
        """Download contents of a single shared folder with progress bar"""
        try:
            folder_path = LOCAL_PATH / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Get complete list of files
            shared_link = dropbox.files.SharedLink(url=shared_url)
            print(f"\nGetting file list for {folder_name}...")
            all_files = get_all_files(dbx, shared_link)
            
            # Filter out already downloaded files
            files_to_download = [
                entry for entry in all_files
                if not (folder_path / entry.name).exists()
            ]
            
            if not files_to_download:
                print(f"{folder_name}: All {len(all_files)} files already downloaded")
                return
            
            print(f"{folder_name}: Downloading {len(files_to_download)} of {len(all_files)} files")
            
            # Create progress bar for this folder
            with tqdm(total=len(files_to_download), desc=f"{folder_name}", unit='file') as pbar:
                for entry in files_to_download:
                    local_file = folder_path / entry.name
                    try:
                        metadata, response = dbx.sharing_get_shared_link_file(
                            url=shared_url,
                            path=f"/{entry.name}"
                        )
                        with open(local_file, "wb") as f:
                            f.write(response.content)
                        pbar.update(1)
                    except dropbox.exceptions.ApiError as e:
                        print(f"\nError downloading {entry.name}: {e}")
                        continue
            
        except dropbox.exceptions.ApiError as e:
            print(f"\nError processing folder {folder_name}: {e}")

    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        LOCAL_PATH.mkdir(exist_ok=True)

        # Download each folder with progress tracking
        for folder_name, url in FOLDER_URLS.items():
            download_folder_contents(dbx, folder_name, url)

        print("\nDownload completed successfully!")
        return str(LOCAL_PATH.absolute())

    except Exception as e:
        print(f"\nError downloading training data: {str(e)}")
        return None