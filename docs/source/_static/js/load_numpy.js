export async function loadNpy(url) {
  let arrayBuffer;
  if(typeof url === "string") {
    const response = await fetch(url);
    arrayBuffer = await response.arrayBuffer();
  }
  else {
    arrayBuffer = await url.arrayBuffer();
  }
  const dataView = new DataView(arrayBuffer);

  // Check magic number
  const magic = Array.from(new Uint8Array(arrayBuffer.slice(0, 6)))
    .map((byte) => String.fromCharCode(byte))
    .join("");
  if (magic !== "\x93NUMPY") {
    throw new Error(`Not a .npy file filename: ${url}`);
  }

  // Parse header
  const headerLength = dataView.getUint16(8, true); // Little endian
  const headerStr = new TextDecoder().decode(
    arrayBuffer.slice(10, 10 + headerLength),
  );
  const header = eval(
    "(" + headerStr.toLowerCase().replace("(", "[").replace(")", "]") + ")",
  );
  const dtype = header.descr;
  const shape = header.shape;

  let data;
  if (dtype === "|u1" || dtype === "|b1") {
    data = new Uint8Array(arrayBuffer, 10 + headerLength);
  } else if (dtype === "<f4") {
    data = new Float32Array(arrayBuffer, 10 + headerLength);
  } else if (dtype === "<f8") {
    data = new Float64Array(arrayBuffer, 10 + headerLength);
  } else if (dtype === "<i4") {
    data = new Int32Array(arrayBuffer, 10 + headerLength);
  } else {
    throw new Error("Unsupported dtype. Only Uint8 is supported. Got" + dtype);
  }
  data.header = header;
  return data;
}
