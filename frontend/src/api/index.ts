import axios from "axios";

const api = axios.create({ baseURL: "http://127.0.0.1:8000/api" });

export async function uploadPdf(file: File) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/upload", form, { headers: { "Content-Type": "multipart/form-data" }});
  return data;
}

export async function getGraphData() {
  try {
    const { data } = await api.get("/graph-data");
    return data;
  } catch (error: any) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data?.detail || "Failed to fetch graph data");
    } else if (error.request) {
      // Request made but no response (backend not running)
      throw new Error("Backend server is not running. Please start the backend server.");
    } else {
      throw new Error(error.message || "Failed to fetch graph data");
    }
  }
}

export async function downloadFile(jobId: string, type: "json"|"excel"|"pdf") {
  const res = await api.get(`/download/${jobId}/${type}`, { responseType: "blob" });
  const url = URL.createObjectURL(res.data);
  const a = document.createElement("a");
  a.href = url;
  a.download = type==="excel" ? "cloudextel-output.xlsx" : `cloudextel-output.${type}`;
  a.click();
  URL.revokeObjectURL(url);
}
