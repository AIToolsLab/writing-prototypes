/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
    theme: {
      extend: {
        colors: {
          sidebar: {
            DEFAULT: '#ffffff',
            hover: '#f1f5f9',
            active: '#e2e8f0',
            text: '#334155',
            border: '#e2e8f0',
          },
          primary: {
            DEFAULT: '#0ea5e9', // Sky 500
            hover: '#0284c7', // Sky 600
            light: '#e0f2fe',
          },
          secondary: {
            DEFAULT: '#94a3b8',
            hover: '#64748b',
          }
        },
        fontFamily: {
          sans: ['Inter', 'sans-serif'],
        }
      },
    },
    plugins: [],
  }

