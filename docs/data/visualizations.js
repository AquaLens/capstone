// PH Map
Plotly.newPlot('ph-map', [{
    type: 'choropleth',
    locationmode: 'country names',
    locations: ['Canada', 'Netherlands', 'India'],
    z: [7.5, 8.2, 6.8],
    colorscale: 'Viridis',
    colorbar: { title: 'pH' }
  }], {
    title: 'Average pH by Country',
    geo: { showframe: false }
  });
  
  // Oxygen Demand Map
  Plotly.newPlot('oxygen-map', [{
    type: 'choropleth',
    locationmode: 'country names',
    locations: ['Mexico', 'USA', 'India'],
    z: [51.2, 20.1, 10.5],
    colorscale: 'YlOrRd',
    colorbar: { title: 'Oxygen Demand (mg/L)' }
  }], {
    title: 'Average Oxygen Demand Levels by Country',
    geo: { showframe: false }
  });
  
  // Nitrogen Map
  Plotly.newPlot('nitrogen-map', [{
    type: 'choropleth',
    locationmode: 'country names',
    locations: ['Greece', 'Italy', 'Poland'],
    z: [11.0, 5.3, 2.1],
    colorscale: 'Electric',
    colorbar: { title: 'Oxidized Nitrogen (mg/L)' }
  }], {
    title: 'Average Oxidized Nitrogen Levels by Country',
    geo: { showframe: false }
  });
  
  // Scatter: Nitrogen vs Oxygen
  Plotly.newPlot('scatter-chart', [{
    x: [0, 2, 4.5, 10.5, 14.5],
    y: [4, 10, 20, 50, 5],
    text: ['Canada', 'France', 'Poland', 'Mexico', 'Greece'],
    mode: 'markers+text',
    type: 'scatter',
    marker: { size: 12, color: 'blue' },
    textposition: 'top center'
  }], {
    title: 'Average Oxidized Nitrogen vs Oxygen Demand',
    xaxis: { title: 'Oxidized Nitrogen (mg/L)' },
    yaxis: { title: 'Oxygen Demand (mg/L)' }
  });
  