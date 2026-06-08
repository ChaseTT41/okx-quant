/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./static/index.html"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg:      '#0a0e14',
        surface: '#141920',
        panel:   '#181d25',
        border:  '#1e2430',
        accent:  '#00d4aa',
        danger:  '#f04770',
      },
    },
  },
  plugins: [],
}
