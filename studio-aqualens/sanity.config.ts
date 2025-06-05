import {defineConfig} from 'sanity'
import {structureTool} from 'sanity/structure'
import {visionTool} from '@sanity/vision'
import {schemaTypes} from './schemaTypes'

export default defineConfig({
  basePath: '/projects/studio',
  name: 'default',
  title: 'AquaLens',

  projectId: '594hcrq0',
  dataset: 'production',

  plugins: [structureTool(), visionTool()],

  schema: {
    types: schemaTypes,
  },

  // 👇 This tells Sanity's Vite bundler to use the correct base path for assets
  vite: {
    base: '/projects/studio/',
  },
})