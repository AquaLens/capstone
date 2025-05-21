import {defineCliConfig} from 'sanity/cli'

export default defineCliConfig({
  api: {
    projectId: '594hcrq0',
    dataset: 'production'
  },
  studioHost: 'aqualens',
  /**
   * Enable auto-updates for studios.
   * Learn more at https://www.sanity.io/docs/cli#auto-updates
   */
  autoUpdates: true,
})
