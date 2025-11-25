import axios from "axios";

const api = axios.create({ baseURL: "http://127.0.0.1:8000/api" });

export async function translateText(text: string, source: string = "mr", target: string = "en") {
  const { data } = await api.post("/translate", { text, source, target });
  return data as { translated_text: string };
}


