import {
  BlobWriter,
  BlobReader,
  ZipReader,
} from "@zip.js/zip.js";
import {TextureLoader} from "three";

const zip_entries = {};
export async function get_file_from_zip(url, filename, return_type = "blob") {
console.log("get_file_from_zip")
  if (zip_entries[url] === undefined) {
    async function get_entries(url) {
      let zipReader;
      if (typeof url === "string") {
        zipReader = new ZipReader(
          new BlobReader(await (await fetch(url)).blob()),
        );
      } else {
        zipReader = new ZipReader(new BlobReader(url));
      }
      const entries = await zipReader.getEntries();
      const entry_map = {};
      for (let entry of entries) {
        console.log(entry.filename);

        entry_map[entry.filename] = entry;
      }
      await zipReader.close();
      return entry_map;
    }
    zip_entries[url] = get_entries(url);
  }
  const entry = (await zip_entries[url])[filename];
  if (!entry) console.error("file", filename, "not found in", url);

      console.log(filename)

  if (entry.filename === filename) {
    const blob = await entry.getData(new BlobWriter());
    if (return_type === "url") return URL.createObjectURL(blob);
    if (return_type === "texture") {
      let t = new TextureLoader().load(URL.createObjectURL(blob));
      console.log(t);
      return t;
    }
    return blob;
  }
}