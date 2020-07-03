#!/usr/bin/env python

import spotipy
import spotipy.oauth2 as oauth2
import config
import pandas as pd
import time
import json
import numpy as np
import requests
import functools
import pickle
from itertools import product


#Work in project conversion OOP.
class Spotify_API_Caller(object):
    """Class that contains common functions for different Spotify API calling use cases"""

    def __init__(self, search_keywords, filename, limit=50, market=None, seed=None):
        """Initiates the API caller object. It takes a list of search term choices and a filename to save backups to,
         Limit caps the amount of results returned in a single search, while market controls the markets parameter."""

        self.wait_min, self.wait_max = 1, 2
        self.start_index, self.current_index = None, None
        self.filename = filename
        self.search_keywords = search_keywords
        self.limit = limit
        self.market = market
        self.skipped, self.missing_features = [], []
        self.removed_duplicates, self.duplicate_count = [], 0
        self.df = pd.DataFrame()
        self.randomizer = np.random.RandomState(seed=seed)
        self.error_count, self.query_times = 0, 0
        self.search_cycles = 0

    def _remove_list_dupes(self, seq):
        """Keeps only unique items in a list, while still preserving order when list size is very large.
        Taken from https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order"""

        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def _spotipy_connector(self):
        """Establishes a connection to the spotify API and returning an object that can make queries."""

        credentials = oauth2.SpotifyClientCredentials(
            client_id=config.spotify_id,
            client_secret=config.spotify_secret)
        self.token = credentials.get_access_token()
        self.conn = spotipy.Spotify(auth=self.token)

    def _manual_spotify_api_call(self, query, type, offset=0):
        """Used for to distinguish between error codes in troubleshooting."""

        market = self.market if self.market else "from_token"
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.token,
        }
        params = (
            ('q', query),
            ('type', type),
            ('limit', self.limit),
            ('offset', offset),
            ('market', market)
        )
        response = requests.get('https://api.spotify.com/v1/search', headers=headers, params=params)
        return response

    def _find_series_unique(self, altered_series, comparison_series, reverse=False):
        """Compares two pandas series objects and returns an array of unique values or duplicates"""

        altered, comparison = altered_series.to_numpy(), comparison_series.to_numpy()
        if reverse:
            dupe_bool = np.in1d(altered, comparison)
            return altered_series[dupe_bool]
        else:
            return np.setdiff1d(altered, comparison)

    def _remove_df_duplicates(self, altered_df, comparison_df, column):
        """Compares a column across two dataframes for duplicates and removes them storing a list of the values
        in an attribute."""

        altered, comparison = altered_df[column], comparison_df[column]
        dupes = self._find_series_unique(altered, comparison, reverse=True)
        orig_count, dupe_count = altered.size, dupes.size
        if dupe_count == 0:
            print(f"Gathered {orig_count} new unique tracks")
            return altered_df
        else:
            print(f"There were {dupe_count} duplicates to skip. Gathered {orig_count-dupe_count} unique tracks")
            self.removed_duplicates.extend(dupes)
            self.removed_duplicates = self._remove_list_dupes(self.removed_duplicates)
            self.duplicate_count += dupe_count
            return altered_df[~altered_df[column].isin(dupes)]


    def _report_missing_features(self, ids, results):
        """Checks list of ids against the ids returned in search results, recording any missing instances to a list."""

        if len(results) > 0:
            comparison = pd.DataFrame(results)['id']
        else:
            comparison = pd.Series()
        missing = self._find_series_unique(ids, comparison)
        self.missing_features.extend(missing)

    def _test_empty_list_2keys(self, query, index, nested_key1, nested_key2):
        """Tests for missing values returning nans if list length is not long enough"""

        try:
            query[index]
        except:
            result = (np.nan, np.nan)
        else:
            result = (query[index][nested_key1], query[index][nested_key2])
        return result

    def _error_increment(self, err_type, func, args):
        """Provides an error message, while adding to the total error count before being routed to the specified function"""

        errors = {"merging":"Retrying last search",
                  "parsing": f"Error in getting data. Cannot parse json. Trying again in {self.wait_time} seconds"}
        print(errors[err_type])
        self.error_count += 1
        return func(*args)

    def _error_checked(func):
        """Decorator used to check called function's total error count before proceeeding"""

        @functools.wraps(func)
        def wrapper_error_check(self, *args, **kwargs):
            if self.error_count >= 5:
                self.error_count = 0
                self.pickle_dump(self)
                raise ValueError('Too many errors. Aborting the API calls')
            else:
                return func(self, *args, **kwargs)
        return wrapper_error_check

    def _spotify_artists_to_dataframe(self, spotify_json, number):
        """Converts spotify artist search to dataframe"""

        df = pd.DataFrame(columns=[f"artist{number}_id", "followers_artist_" + number, "popularity_artist_" + number,
                                   "genres_artist_" + number])
        if spotify_json == "skipped":
            print(f"Using empty dataframe for artists {number}")
        else:
            for item in spotify_json["artists"]:
                id, followers = item["id"], item["followers"]["total"]
                popularity, genres = item["popularity"], item["genres"]
                columns = {f"artist{number}_id": id, "followers_artist_" + number: followers,
                           "popularity_artist_" + number: popularity, "genres_artist_" + number: genres}
                df = df.append(columns, ignore_index=True)
        df[f"artist{number}_id"].drop_duplicates(inplace=True)
        return df

    def _spotify_audio_features_to_dataframe(self, spotify_json):

        if len(spotify_json) == 0 or spotify_json == "skipped":
            print("Using empty dataframe for audio features")
            return pd.DataFrame(columns=["id"])
        else:
            return pd.DataFrame(spotify_json)


    def _spotify_search_to_dataframe(self, spotify_json, structure="track_search"):
        """Converts spotify search json to dataframe"""

        def spotify_search_to_df_line(df, item):
            """"There are minor differences in certain jsons returned by Spotify. This function maintains consistency
            within the inner parameters."""

            track_id, track_name = item["id"], item["name"]
            popularity, markets = item["popularity"], item["available_markets"]
            track_number = item["track_number"]
            album_name, album_id, album_release = item["album"]["name"], item["album"]["id"], item["album"][
                "release_date"]
            artist1_name, artist1_id = item["artists"][0]["name"], item["artists"][0]["id"]
            artist2_name, artist2_id = self._test_empty_list_2keys(item["artists"], 1, "name", "id")
            duration, explicit = item["duration_ms"], item["explicit"]
            columns = {"id": track_id, "name": track_name, "popularity": popularity, \
                       "artist1_name": artist1_name, "artist1_id": artist1_id, \
                       "artist2_name": artist2_name, "artist2_id": artist2_id, \
                       "album_name": album_name, "album_id": album_id, "track_number": track_number, \
                       "release_date": album_release, "duration": duration, "explicit": explicit, "markets": markets}
            return df.append(columns, ignore_index=True)

        df = pd.DataFrame(columns=["id", "name", "popularity", "artist1_name", "artist1_id", \
                                   "artist2_name", "artist2_id", "album_name", "album_id", "release_date", \
                                   "track_number", "duration", "explicit", "markets"])

        if structure == "track_search":
            for item in spotify_json["tracks"]["items"]:
                df = spotify_search_to_df_line(df, item)

        elif structure == "playlist_tracks":
            for item in spotify_json["items"]:
                track = item["track"]
                df = spotify_search_to_df_line(df, track)

        else:
            print("Unknown structure argument, Returning empty df.")

        return df

    def _wait_cycle(self):
        """Randomly chooses a wait time based on the min and max values and waits for that duration."""

        self.wait_time = np.random.uniform(self.wait_min, self.wait_max)
        time.sleep(self.wait_time)

    def pickle_dump(self, data):
        """Saves a backup pickle of the scrapped data appending the interval numbers of the data that was taken."""

        with open(self.filename + f"{self.start_index}-{self.current_index}.pickle", 'wb') as f:
            pickle.dump(data, f)

    @_error_checked
    def spotify_find_by_ids(self, df, column_name, search_type):
        """Takes a dataframe of Spotify search results and gets details of audio features or artists"""

        self._wait_cycle()
        ids = df[column_name].dropna()
        if ids.size == 0:
            print(f"There are no ids for {column_name}. Skipping")
            return "skipped"
        try:
            if search_type == "features":
                print("Searching audio features")
                search = self.conn.audio_features(ids)
                search = list(filter(lambda x: x != None, search))
                self._report_missing_features(ids, search)
            elif search_type == "artists":
                print(f"Searching {column_name}")
                search = self.conn.artists(ids)
            else:
                print("Choose a supported search function")
                search = None
        except:
            return self._error_increment("parsing", self.spotify_find_by_ids, [df, column_name, search_type])
        else:
            return search


class Spotify_General_Track_Searcher(Spotify_API_Caller):
    """Used to do a blanket search across all tracks in the Spotify API"""

    def __init__(self, search_keywords, filename, limit=50, market=None, seed=None, start=0, end=40, samples=None):
        """Takes the variables of the parent class along with others that control the creation of the random
        or complete search list."""

        super().__init__(search_keywords, filename, limit, market, seed)
        self._generate_search_list(start, end, samples)

    def _generate_search_list(self, start, end, samples):
        """Generates a list of items to be searched in the api call. If a number of samples are specified, it generates
        a list of random terms/offsets. If not, it takes all terms adding all offsets for each."""
        if samples:
            self.search_list = self._pseudo_random_terms(start, end, samples)
        else:
            span = list(np.arange(start, end))
            joined = [self.search_keywords, span]
            self.search_list = list(product(*joined))
        self.search_list_size = len(self.search_list)
        self.orig_search_list = False

    def _pseudo_random_terms(self, start, end, samples):
        """Creates tuples of random_search combos to get more results from capped searches."""

        offsets = self.randomizer.randint(start, end, size=samples)
        choices = self.randomizer.choice(self.search_keywords, size=samples)
        random_list = list(zip(choices, offsets))
        return self._remove_list_dupes(random_list)


    def _merge_spotify_search_results(self, main_df, audio_features, artist1_search, artist2_search):
        """"Takes the individual search results merging them into a single Dataframe"""

        audio_feat_df = self._spotify_audio_features_to_dataframe(audio_features)
        artist1_df = self._spotify_artists_to_dataframe(artist1_search, "1")
        artist2_df = self._spotify_artists_to_dataframe(artist2_search, "2")
        # Combines two searches into a single dataframe
        try:
            joined_df = pd.merge(main_df, audio_feat_df, how="left", on="id")
            joined_df = pd.merge(joined_df, artist1_df, how="left", on="artist1_id")
            joined_df = pd.merge(joined_df, artist2_df, how="left", on="artist2_id")
        except:
            print("Error merging dataframes.")
            return False
        else:
            return joined_df

    def _search_list_stepper(self, counter):
        """Sets current search terms, offsets, and indices based on position in the search list."""

        self.search_term = self.search_list[counter][0]
        self.offset = self.search_list[counter][1] * self.limit
        self.current_index = f"{self.search_cycles:02d}_{counter:03d}"
        if not self.start_index:
            self.start_index = self.current_index

    def _search_list_skipped_swap(self):
        """Swaps the search list with the skipped list and increments the search cycle counter"""

        self.search_cycles += 1
        self.search_list = self.skipped
        self.skipped = []
        self.search_list_size = len(self.search_list)
        self.query_times = 0

    def _search_list_verify(self):
        """Evaluates whether or not to switch the search list with what has been skipped."""

        skipped_count, missing_feature_count = len(self.skipped), len(self.missing_features)
        if skipped_count == 0 and missing_feature_count == 0:
            print("No skipped searches or missing features. Search complete.")
        elif skipped_count == 0:
            print(f"No skipped searches, but {missing_feature_count} features are missing.")
        elif self.orig_search_list and skipped_count == self.search_list_size:
            print(f"After searching again, there were also {skipped_count} skipped items. Try searching for the"
                  f"{missing_feature_count} missing features")
        elif self.orig_search_list:
            print(f"Found {self.search_list_size-skipped_count} new items, but {skipped_count} are still missing."
                  f"Continue searching")
            self._search_list_skipped_swap()
        else:
            print(f"There were {skipped_count} items missing. Continue search.")
            self.orig_search_list = self.search_list
            self._search_list_skipped_swap()

    @Spotify_API_Caller._error_checked
    def spotify_search(self, query, offset=0, search_type="track"):
        """Searches Spotify's API for tracks"""

        self._wait_cycle()
        try:
            search = self.conn.search(query, type=search_type, offset=offset, limit=self.limit, market=self.market)
        except spotipy.client.SpotifyException:
            test = self._manual_spotify_api_call(query, search_type, offset)
            if test.status_code == 404:
                print("Result does not exist. Recording to error log and skipping to the next")
                return "skip"
            else:
                return self._error_increment("parsing", self.spotify_search, [query, offset, search_type])

        else:
            if search["tracks"]["items"] == None or search["tracks"]["items"] == []:
                print("Result does not exist. Recording to error log and skipping to the next")
                return "skip"
        return search


    def spotify_read_loop(self, main_path, features_ext, artist1_ext, artist2_ext, last_item, first_item=0):
        """This function is now obsolete. This would read json backups and construct a dataframe from those before
        the overall backup functionality of the API call object was completed."""

        master_df = pd.DataFrame()
        self.start_index = f"{first_item:03d}"
        for i in np.arange(first_item, last_item + 1):
            self.current_index = f"{i:03d}"
            ind = self.current_index
            joined_path = main_path + ind
            try:
                search = json_read(joined_path)
            except:
                print(f"{joined_path} does not exist moving to the next")
                continue
            else:
                audio_features_search = json_read(main_path + ind + features_ext + ind)
                artist1_search = json_read(main_path + ind + artist1_ext + ind)
                artist2_search = json_read(main_path + ind + artist2_ext + ind)
            merged = self._merge_spotify_search_results(search, audio_features_search, artist1_search, artist2_search)
            master_df = master_df.append(merged)
        self.df = master_df
        self.pickle_dump(self)

    @Spotify_API_Caller._error_checked
    def spotify_search_loop(self, stop_point=False):
        """Combines the Spotify search functions looping through API calls handling exceptions when needed. Specify a stop
        point if you would like to end the search/save a backup before the entire search list is completed"""

        # Establishes connection to spotify API
        self._spotipy_connector()

        # Initializes a while loop and establishes the weight time in between queries if the script has completed at least 1 loop.
        while self.query_times < self.search_list_size:

            # Checks to see if there is a stop point and if the amount of searches has reached it. If so, it backs up the object.
            # If not, it waits before continuing the search,
            if stop_point and stop_point <= self.query_times:
                print(
                    f"Reached stop point of {stop_point}. There are {self.search_list_size - stop_point} remaining searches")
                self.pickle_dump(self)
                return self.df
            else:
                self._wait_cycle()

            # Generates class attributes for the current iteration in the search loop based on position in the search list.
            self._search_list_stepper(self.query_times)

            # Conducts a search some of those attributes.
            search = self.spotify_search(self.search_term, self.offset, "track")

            # If the search function decided to move past the current search, it saves the missing term/offset to a list to be
            # evaluated later.
            if search == "skip":
                self.skipped.append((self.search_term, self.offset))
                self.query_times += 1
                return self.spotify_search_loop(stop_point)

            else:
                # Prints a bit of the text to provide a status update and parses it as a json.
                print(f"parsing {self.query_times} of {self.search_list_size}")

            # Tests the results to see that it is in the proper format. If it doesn't work, it calls the function from where it left off.
            try:
                search['tracks']
            except:
                return self._error_increment("parsing", self.spotify_search_loop, [stop_point])

            # Captures the output into a dataframe and dumps a json file.
            finally:
                json_dump(search, self.filename, self.current_index)

                # Insert function here that formats search results into a pandas DataFrame
                main_df = self._spotify_search_to_dataframe(search)
                main_df = self._remove_df_duplicates(main_df, self.df, "id")

            # Conducts a search using the id's from the dataframe for audio features.
            audio_features = self.spotify_find_by_ids(main_df, "id", "features")
            json_dump(audio_features, self.filename + self.current_index + "audio_features", self.current_index)

            # Conducts searches for extra artist info:
            artist1_search = self.spotify_find_by_ids(main_df, "artist1_id", "artists")
            artist2_search = self.spotify_find_by_ids(main_df, "artist2_id", "artists")

            json_dump(artist1_search, self.filename + self.current_index + "artist1_search", self.current_index)
            json_dump(artist2_search, self.filename + self.current_index + "artist2_search", self.current_index)

            # Merges all results together.
            merged = self._merge_spotify_search_results(main_df, audio_features, artist1_search, artist2_search)
            if type(merged) != pd.core.frame.DataFrame:
                return self._error_increment("merging", self.spotify_search_loop, [stop_point])
            else:
                self.df = self.df.append(merged, ignore_index=True)

            # increments and continues on to the next api call
            self.query_times += 1
            self.error_count = 0

        self._search_list_verify()
        self.pickle_dump(self)
        return self.df


class Spotify_Playlist_Scraper(Spotify_API_Caller):
    def __init__(self, search_keywords, filename, limit=50, market=None, seed=None):
        """Gathers data from spotify playlists"""

        super().__init__(search_keywords, filename, limit, market, seed)
        self.playlist_tracks = pd.DataFrame()

    @Spotify_API_Caller._error_checked
    def spotify_find_by_single_id(self, id, search_type, offset=0):
        """Searches the spotify api for calls that take a single id as the parameter"""

        try:
            if search_type == "playlist_tracks":
                print("Getting playlist tracks")
                search = self.conn.playlist_tracks(id, limit=self.limit, offset=offset, market=self.market)
            elif search_type == "artist_albums":
                print("Getting artist albums")
                search = self.conn.artist_albums(id, limit=self.limit, offset=offset)
            elif search_type == "album_tracks":
                print("Getting artist albums")
                search = self.conn.artist_albums(id, limit=self.limit, offset=offset)
            else:
                print("Choose a supported search function")
                search = None
        except:
            return self._error_increment("parsing", self.spotify_find_by_single_id, [id, search_type, offset])
        else:
            return search

    def spotify_get_playlist_tracks(self, offset):
        self._spotipy_connector()
        search_type = "playlist_tracks"
        next = True
        while next:
            self._wait_cycle()
            print(f"getting tracks {offset} - {offset+self.limit}")
            search = self.spotify_find_by_single_id(self.search_keywords, search_type, offset)
            df = self._spotify_search_to_dataframe(search, search_type)
            self.playlist_tracks = self.playlist_tracks.append(df)
            offset += self.limit
            next = search["next"]



#Reads a json path.
def json_read(path):
    with open(path+".json") as f:
        return json.load(f)

# Saves a backup json of the scrapped data.
def json_dump(search_results, filename, query_times):
    with open(filename + f"{query_times}.json", 'w') as json_file:
        json.dump(search_results, json_file)


def pickle_read(path):
    with open(path, "rb") as f:
        pickle_file = pickle.load(f)
    return pickle_file

