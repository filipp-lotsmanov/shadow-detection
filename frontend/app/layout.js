import "./globals.css";

export const metadata = {
  title: "Shadow Detection",
  description: "Predict off-screen pedestrian locations from shadow imagery.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
