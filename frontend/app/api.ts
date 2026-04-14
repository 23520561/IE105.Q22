export async function getData<T>(url: string): Promise<T | null> {
  try {
    const response = await fetch(url, {
      method: "GET",
    });
    if (!response.ok) {
      throw new Error(`Response status: ${response.status}`);
    }
    return await response.json();
  } catch (err: unknown) {
    if (err instanceof Error) {
      console.error(err.message);
    } else {
      console.error(err);
    }
    return null;
  }
}
export const apiUrl = import.meta.env.VITE_API_URL;
