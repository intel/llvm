// Copyright (C) 2024-2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core state
let activeRuns = new Set(defaultCompareNames);
let chartInstances = new Map();
let suiteNames = new Set();
let activeTags = new Set();
let timeseriesData, barChartsData, allRunNames;
let layerComparisonsData;
let latestRunsLookup = new Map();
let pendingCharts = new Map(); // Store chart data for lazy loading
let chartObserver; // Intersection observer for lazy loading charts
let annotationsOptions = new Map(); // Global options map for annotations
let archivedDataLoaded = false;
let loadedBenchmarkRuns = []; // Loaded results from the js/json files
let isInitializing = true; // Flag for a proper handling of URLs
// Global variables loaded from data.js/data.json:
// - benchmarkRuns: array of benchmark run data
// - benchmarkMetadata: metadata for benchmarks and groups
// - benchmarkTags: tag definitions
// - defaultCompareNames: default run names for comparison
// - flamegraphData: available flamegraphs data with runs information (if available)

// Helper function to get base URL for remote or local resources
function getResourceBaseUrl() {
    return typeof remoteDataUrl !== 'undefined' && remoteDataUrl !== ''
        ? 'https://raw.githubusercontent.com/intel/llvm-ci-perf-results/unify-ci'
        : '.';
}

// Toggle configuration and abstraction
//
// HOW TO ADD A NEW TOGGLE:
// 1. Add HTML checkbox to index.html:
//    <label><input type="checkbox" id="my-toggle">My Toggle</label>
//
// 2. Add configuration below:
//    'my-toggle': {
//        defaultValue: false,              // true = enabled by default, false = disabled by default
//        urlParam: 'myParam',              // Name shown in URL (?myParam=true)
//        invertUrlParam: false,            // false = normal behavior, true = legacy inverted logic
//        onChange: function(isEnabled) {  // Function called when toggle state changes
//            // Your logic here
//            updateURL();                  // Always call this to update the browser URL
//        }
//    }
//
// 3. (Optional) Add helper function for cleaner, more readable code:
//    function isMyToggleEnabled() { return isToggleEnabled('my-toggle'); }
//
//    This lets you write: if (isMyToggleEnabled()) { ... }
//    Instead of:         if (isToggleEnabled('my-toggle')) { ... }
//

const toggleConfigs = {
    'show-notes': {
        defaultValue: true,
        urlParam: 'notes',
        invertUrlParam: true, // Store false in URL when enabled (legacy behavior)
        onChange: function (isEnabled) {
            document.querySelectorAll('.benchmark-note').forEach(note => {
                note.style.display = isEnabled ? 'block' : 'none';
            });
            if (!isInitializing) {
                updateURL();
            }
        }
    },
    'show-unstable': {
        defaultValue: false,
        urlParam: 'unstable',
        invertUrlParam: false,
        onChange: function (isEnabled) {
            document.querySelectorAll('.benchmark-unstable').forEach(warning => {
                warning.style.display = isEnabled ? 'block' : 'none';
            });
            filterCharts();
        }
    },
    'custom-range': {
        defaultValue: false,
        urlParam: 'customRange',
        invertUrlParam: false,
        onChange: function (isEnabled) {
            updateCharts();
        }
    },
    'show-archived-data': {
        defaultValue: false,
        urlParam: 'archived',
        invertUrlParam: false,
        onChange: function (isEnabled) {
            if (isEnabled) {
                loadArchivedData();
            } else {
                if (archivedDataLoaded) {
                    location.reload();
                }
            }
            if (!isInitializing) {
                updateURL();
            }
        }
    }
    ,
    'show-flamegraph': {
        defaultValue: false,
        urlParam: 'flamegraph',
        invertUrlParam: false,
        onChange: function (isEnabled) {
            // Toggle between flamegraph-only display and normal charts
            updateCharts();
            updateFlameGraphTooltip();
            // Refresh download buttons to adapt to new mode
            refreshDownloadButtons();
            if (!isInitializing) {
                updateURL();
            }
        }
    }
};

// Generic toggle helper functions
function isToggleEnabled(toggleId) {
    const toggle = document.getElementById(toggleId);
    return toggle ? toggle.checked : toggleConfigs[toggleId]?.defaultValue || false;
}

function setupToggle(toggleId, config) {
    const toggle = document.getElementById(toggleId);
    if (!toggle) return;

    // Set up event listener
    toggle.addEventListener('change', function () {
        config.onChange(toggle.checked);
    });

    // Initialize from URL params if present
    const urlParam = getQueryParam(config.urlParam);
    if (urlParam !== null) {
        const urlValue = urlParam === 'true';
        // Handle inverted URL params (like notes where false means enabled)
        toggle.checked = config.invertUrlParam ? !urlValue : urlValue;
    } else {
        // Use default value
        toggle.checked = config.defaultValue;
    }

    // Ensure the initial toggle state is applied to the UI immediately
    // (important after merges where defaults or URL params determine initial view)
    try {
        config.onChange(toggle.checked);
    } catch (e) {
        // Swallow errors from onChange during initialization to avoid breaking
        // the whole page load; developers can investigate specific toggle handlers.
        console.error(`Error while applying initial state for toggle ${toggleId}:`, e);
    }
}

function updateToggleURL(toggleId, config, url) {
    const isEnabled = isToggleEnabled(toggleId);

    if (config.invertUrlParam) {
        // For inverted params, store in URL when disabled
        if (isEnabled) {
            url.searchParams.delete(config.urlParam);
        } else {
            url.searchParams.set(config.urlParam, 'false');
        }
    } else {
        // For normal params, store in URL when enabled
        if (!isEnabled) {
            url.searchParams.delete(config.urlParam);
        } else {
            url.searchParams.set(config.urlParam, 'true');
        }
    }
}


// DOM Elements
let runSelect, selectedRunsDiv, suiteFiltersContainer, tagFiltersContainer;

// Observer for lazy loading charts
function initChartObserver() {
    if (chartObserver) return;

    chartObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const containerId = entry.target.querySelector('canvas').id;
                if (pendingCharts.has(containerId)) {
                    const { data, type } = pendingCharts.get(containerId);
                    createChart(data, containerId, type);
                    pendingCharts.delete(containerId);
                    chartObserver.unobserve(entry.target);
                }
            }
        });
    }, {
        root: null, // viewport (current view)
        rootMargin: '100px', // Load charts a bit before they enter the viewport
        threshold: 0.1 // Start loading when 10% of the chart is within the rootMargin
    });
}

const colorPalette = [
    'rgb(255, 50, 80)',
    'rgb(255, 145, 15)',
    'rgb(255, 220, 0)',
    'rgb(20, 200, 50)',
    'rgb(0, 130, 255)',
    'rgb(180, 60, 255)',
    'rgb(255, 40, 200)',
    'rgb(0, 210, 180)',
    'rgb(255, 90, 0)',
    'rgb(110, 220, 0)',
    'rgb(240, 100, 170)',
    'rgb(30, 175, 255)',
    'rgb(180, 210, 0)',
    'rgb(130, 0, 220)',
    'rgb(255, 170, 0)',
    'rgb(0, 170, 110)',
    'rgb(220, 80, 60)',
    'rgb(80, 115, 230)',
    'rgb(210, 190, 0)',
];

const annotationPalette = [
    'rgba(167, 109, 59, 0.8)',
    'rgba(185, 185, 60, 0.8)',
    'rgba(58, 172, 58, 0.8)',
    'rgba(158, 59, 158, 0.8)',
    'rgba(167, 93, 63, 0.8)',
    'rgba(163, 60, 81, 0.8)',
    'rgba(51, 148, 155, 0.8)',
]

const nameColorMap = {};
let colorIndex = 0;

function getColorForName(name) {
    if (!(name in nameColorMap)) {
        nameColorMap[name] = colorPalette[colorIndex % colorPalette.length];
        colorIndex++;
    }
    return nameColorMap[name];
}

// Run selector functions
function updateSelectedRuns(forceUpdate = true) {
    selectedRunsDiv.innerHTML = '';
    activeRuns.forEach(name => {
        selectedRunsDiv.appendChild(createRunElement(name));
    });

    // Update platform information for selected runs
    displaySelectedRunsPlatformInfo();

    if (forceUpdate)
        updateCharts();
}

function createRunElement(name) {
    const runElement = document.createElement('span');
    runElement.className = 'selected-run';
    runElement.innerHTML = `${name} <button onclick="removeRun('${name}')">X</button>`;
    return runElement;
}

function addSelectedRun() {
    if (!runSelect) return; // Safety check for DOM element
    const selectedRun = runSelect.value;
    if (selectedRun && !activeRuns.has(selectedRun)) {
        activeRuns.add(selectedRun);
        updateSelectedRuns();
    }
}

function removeRun(name) {
    activeRuns.delete(name);
    updateSelectedRuns();
}

// Chart creation and update
function createChart(data, containerId, type) {
    if (chartInstances.has(containerId)) {
        chartInstances.get(containerId).destroy();
    }

    const ctx = document.getElementById(containerId).getContext('2d');

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: true,
                text: data.display_label || data.label
            },
            subtitle: {
                display: true,
                text: data.lower_is_better ? "Lower is better" : "Higher is better"
            },
            tooltip: {
                callbacks: {
                    label: (context) => {
                        if (type === 'time') {
                            const point = context.raw;
                            return [
                                `${point.seriesName}:`,
                                `Value: ${point.y.toFixed(2)} ${data.unit}`,
                                `Stddev: ${point.stddev.toFixed(2)} ${data.unit}`,
                                `Git Hash: ${point.gitHash}`,
                                ...(point.compute_runtime && point.compute_runtime !== 'null' && point.compute_runtime !== 'unknown' ?
                                    [`Compute Runtime: ${point.compute_runtime}`] : []),
                                ...(point.gitBenchHash ? [`Bench hash: ${point.gitBenchHash.substring(0, 7)}`] : []),
                                ...(point.gitBenchUrl ? [`Bench URL: ${point.gitBenchUrl}`] : []),
                            ];
                        } else {
                            return [`${context.dataset.label}:`,
                            `Value: ${context.parsed.y.toFixed(2)} ${data.unit}`,
                            ];
                        }
                    }
                }
            },
            legend: {
                position: 'top',
                labels: {
                    boxWidth: 12,
                    padding: 10,
                }
            },
            annotation: type === 'time' ? {
                annotations: {}
            } : undefined
        },
        scales: {
            y: {
                title: {
                    display: true,
                    text: data.unit
                },
                grace: '20%',
                min: isCustomRangesEnabled() ? data.range_min : null,
                max: isCustomRangesEnabled() ? data.range_max : null
            }
        }
    };

    if (type === 'time') {
        options.interaction = {
            mode: 'nearest',
            intersect: true // Require to hover directly over a point
        };
        options.onClick = (event, elements) => {
            if (elements.length > 0) {
                const point = elements[0].element.$context.raw;
                if (point.gitHash && point.gitRepo) {
                    window.open(`${point.gitRepo}/commit/${point.gitHash}`, '_blank');
                }
            }
        };
        options.scales.x = {
            type: 'timeseries',
            time: {
                unit: 'day'
            },
            ticks: {
                maxRotation: 45,
                minRotation: 45,
                autoSkip: true,
                maxTicksLimit: 10
            }
        };

        // Add dependencies version change annotations
        if (Object.keys(data.runs).length > 0) {
            ChartAnnotations.addVersionChangeAnnotations(data, options);
        }
    }

    const chartConfig = {
        type: type === 'time' ? 'line' : 'bar',
        data: type === 'time' ? {
            datasets: Object.values(data.runs).map(runData => ({
                ...runData,
                // For timeseries (historical results charts) use runName,
                // otherwise use displayLabel (for layer comparison charts)
                label: containerId.startsWith('timeseries') ?
                    runData.runName :
                    (runData.displayLabel || runData.label)
            }))
        } : {
            labels: data.labels,
            datasets: data.datasets
        },
        options: options
    };

    const chart = new Chart(ctx, chartConfig);
    chartInstances.set(containerId, chart);

    // Set explicit canvas size after chart creation to ensure proper sizing
    const canvas = document.getElementById(containerId);
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Calculate dynamic height based on number of legend items
    const legendItemCount = type === 'time' ?
        Object.values(data.runs).length :
        data.datasets.length;

    // Base chart height + legend height (25px per line + padding)
    const baseChartHeight = 350;
    const legendHeight = Math.max(legendItemCount * 25, 50); // minimum 50px for legend
    const totalHeight = baseChartHeight + legendHeight;

    // Set canvas dimensions for crisp rendering
    canvas.width = rect.width * dpr;
    canvas.height = totalHeight * dpr;

    // Scale the context to ensure correct drawing operations
    const context = canvas.getContext('2d');
    context.scale(dpr, dpr);

    // Force chart to use these exact dimensions
    chart.resize(rect.width, totalHeight);

    // Add annotation interaction handlers for time-series charts
    if (type === 'time') {
        ChartAnnotations.setupAnnotationListeners(chart, ctx, options);
    }

    return chart;
}

function updateCharts() {
    const filterRunData = (chart) => ({
        ...chart,
        runs: Object.fromEntries(
            Object.entries(chart.runs).filter(([_, data]) =>
                activeRuns.has(data.runName)
            )
        )
    });

    const filteredTimeseriesData = timeseriesData.map(filterRunData);
    const filteredLayerComparisonsData = layerComparisonsData.map(filterRunData);

    const filteredBarChartsData = barChartsData.map(chart => ({
        ...chart,
        labels: chart.labels.filter(label => activeRuns.has(label)),
        datasets: chart.datasets.map(dataset => ({
            ...dataset,
            data: dataset.data.filter((_, i) => activeRuns.has(chart.labels[i]))
        }))
    }));

    drawCharts(filteredTimeseriesData, filteredBarChartsData, filteredLayerComparisonsData);
}

// Function to refresh download buttons when mode changes
function refreshDownloadButtons() {
    // Wait a bit for charts to be redrawn
    setTimeout(() => {
        document.querySelectorAll('.chart-download-button').forEach(button => {
            const container = button.closest('.chart-container');
            if (container && button.updateChartButton) {
                button.updateChartButton();
            }
        });
    }, 100);
}

function drawCharts(filteredTimeseriesData, filteredBarChartsData, filteredLayerComparisonsData) {
    // Clear existing charts
    document.querySelectorAll('.charts').forEach(container => container.innerHTML = '');
    chartInstances.forEach(chart => chart.destroy());
    chartInstances.clear();
    pendingCharts.clear();

    initChartObserver(); // For lazy loading charts

    // Create timeseries charts
    filteredTimeseriesData.forEach((data, index) => {
        const containerId = `timeseries-${index}`;
        const container = createChartContainer(data, containerId, 'benchmark');
        document.querySelector('.timeseries .charts').appendChild(container);

        // Only set up chart observers if not in flamegraph mode
        if (!isFlameGraphEnabled()) {
            pendingCharts.set(containerId, { data, type: 'time' });
            chartObserver.observe(container);
        }
    });

    // Create layer comparison charts
    filteredLayerComparisonsData.forEach((data, index) => {
        const containerId = `layer-comparison-${index}`;
        const container = createChartContainer(data, containerId, 'group');
        document.querySelector('.layer-comparisons .charts').appendChild(container);

        // Only set up chart observers if not in flamegraph mode
        if (!isFlameGraphEnabled()) {
            pendingCharts.set(containerId, { data, type: 'time' });
            chartObserver.observe(container);
        }
    });

    // Create bar charts
    filteredBarChartsData.forEach((data, index) => {
        const containerId = `barchart-${index}`;
        const container = createChartContainer(data, containerId, 'group');
        document.querySelector('.bar-charts .charts').appendChild(container);

        // Only set up chart observers if not in flamegraph mode
        if (!isFlameGraphEnabled()) {
            pendingCharts.set(containerId, { data, type: 'bar' });
            chartObserver.observe(container);
        }
    });

    // Apply current filters
    filterCharts();
}

function createChartContainer(data, canvasId, type) {
    const container = document.createElement('div');
    container.className = 'chart-container';
    container.setAttribute('data-label', data.label);
    container.setAttribute('data-suite', data.suite);

    // Create header section for metadata
    const headerSection = document.createElement('div');
    headerSection.className = 'chart-header';

    // Check if this benchmark is marked as unstable
    const metadata = metadataForLabel(data.label, type);
    if (metadata && metadata.unstable) {
        container.setAttribute('data-unstable', 'true');

        // Add unstable warning
        const unstableWarning = document.createElement('div');
        unstableWarning.className = 'benchmark-unstable';
        unstableWarning.textContent = metadata.unstable;
        unstableWarning.classList.toggle('hidden', !isUnstableEnabled());
        headerSection.appendChild(unstableWarning);
    }

    // Add description if present in metadata
    if (metadata && metadata.description) {
        const descElement = document.createElement('div');
        descElement.className = 'benchmark-description';
        descElement.textContent = metadata.description;
        headerSection.appendChild(descElement);
    }

    // Add notes if present
    if (metadata && metadata.notes) {
        const noteElement = document.createElement('div');
        noteElement.className = 'benchmark-note';
        noteElement.textContent = metadata.notes;
        noteElement.classList.toggle('hidden', !isNotesEnabled());
        headerSection.appendChild(noteElement);
    }

    // Add tags if present
    if (metadata && metadata.tags) {
        container.setAttribute('data-tags', metadata.tags.join(','));

        // Add tags display
        const tagsContainer = document.createElement('div');
        tagsContainer.className = 'benchmark-tags';

        metadata.tags.forEach(tag => {
            const tagElement = document.createElement('span');
            tagElement.className = 'tag';
            tagElement.textContent = tag;
            tagElement.setAttribute('data-tag', tag);

            // Add tooltip with tag description
            if (benchmarkTags[tag]) {
                tagElement.setAttribute('title', benchmarkTags[tag].description);
            }

            tagsContainer.appendChild(tagElement);
        });

        headerSection.appendChild(tagsContainer);
    }

    // Add header section to container
    container.appendChild(headerSection);

    // Create main content section (chart + legend area)
    const contentSection = document.createElement('div');
    contentSection.className = 'chart-content';

    // Check if flamegraph mode is enabled
    if (isFlameGraphEnabled()) {
        // Get all flamegraph data for this benchmark from selected runs
        const flamegraphsToShow = getFlameGraphsForBenchmark(data.label, activeRuns);

        if (flamegraphsToShow.length > 0) {
            // Add a class to reduce padding for flamegraph containers
            container.classList.add('flamegraph-chart');
            // Create individual containers for each flamegraph to give them proper space
            flamegraphsToShow.forEach((flamegraphInfo, index) => {
                // Create a dedicated container for this flamegraph
                const flamegraphContainer = document.createElement('div');
                flamegraphContainer.className = 'flamegraph-container';

                const iframe = document.createElement('iframe');
                iframe.src = flamegraphInfo.path;
                iframe.className = 'flamegraph-iframe';
                iframe.title = `${flamegraphInfo.runName} - ${data.label}`;

                // Add error handling for missing flamegraph files
                iframe.onerror = function () {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'flamegraph-error';
                    errorDiv.textContent = `No flamegraph available for ${flamegraphInfo.runName} - ${data.label}`;
                    flamegraphContainer.replaceChild(errorDiv, iframe);
                };

                flamegraphContainer.appendChild(iframe);
                contentSection.appendChild(flamegraphContainer);
            });

            // No need for resize handling since CSS handles all sizing
            // The flamegraphs will automatically use the full container width
        } else {
            // Show message when no flamegraph is available
            const noFlameGraphDiv = document.createElement('div');
            noFlameGraphDiv.className = 'flamegraph-unavailable';
            noFlameGraphDiv.textContent = `No flamegraph data available for ${data.label}`;
            contentSection.appendChild(noFlameGraphDiv);
        }
    } else {
        // Canvas for the chart - fixed position in content flow
        const canvas = document.createElement('canvas');
        canvas.id = canvasId;
        canvas.className = 'benchmark-canvas';
        contentSection.appendChild(canvas);
    }

    container.appendChild(contentSection);

    // Add simple flamegraph links below the chart: left label, inline orange links
    (function addFlamegraphLinks() {
        try {
            const flamegraphs = getFlameGraphsForBenchmark(data.label, activeRuns || new Set());
            if (!flamegraphs || flamegraphs.length === 0) return;

            const outer = document.createElement('div');
            outer.className = 'chart-flamegraph-links';

            const label = document.createElement('div');
            label.className = 'flamegraph-label';
            label.textContent = 'Flamegraph(s):';

            const links = document.createElement('div');
            links.className = 'flamegraph-links-inline';

            flamegraphs.forEach(fg => {
                const a = document.createElement('a');
                a.className = 'flamegraph-link';
                a.href = fg.path;
                a.target = '_blank';

                // flame emoticon before run name
                const icon = document.createElement('span');
                icon.className = 'flame-icon';
                icon.textContent = '🔥';

                const text = document.createElement('span');
                text.className = 'flame-text';
                text.textContent = fg.runName ? `${fg.runName}${fg.timestamp ? ' — ' + fg.timestamp : ''}` : (fg.timestamp || 'Flamegraph');

                a.appendChild(icon);
                a.appendChild(text);
                links.appendChild(a);
            });

            outer.appendChild(label);
            outer.appendChild(links);
            container.appendChild(outer);
        } catch (e) {
            console.error('Error while adding flamegraph links for', data.label, e);
        }
    })();

    // Create footer section for details
    const footerSection = document.createElement('div');
    footerSection.className = 'chart-footer';

    // Create details section for extra info
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.className = 'download-summary';
    summary.textContent = "Details";

    // Helper: format Date to YYYYMMDD_HHMMSS (UTC)
    function formatTimestampFromDate(d) {
        if (!d) return null;
        const date = (d instanceof Date) ? d : new Date(d);
        if (isNaN(date)) return null;
        const pad = (n) => n.toString().padStart(2, '0');
        const Y = date.getUTCFullYear();
        const M = pad(date.getUTCMonth() + 1);
        const D = pad(date.getUTCDate());
        const h = pad(date.getUTCHours());
        const m = pad(date.getUTCMinutes());
        const s = pad(date.getUTCSeconds());
        return `${Y}${M}${D}_${h}${m}${s}`;
    }

    // Base raw URL for archives (branch-based)
    const RAW_BASE = getResourceBaseUrl();

    // Helper function to show flamegraph list
    function showFlameGraphList(flamegraphs, buttonElement) {
        const existingList = document.querySelector('.download-list');
        if (existingList) existingList.remove();

        const listContainer = document.createElement('div');
        listContainer.className = 'download-list';

        // Dynamic positioning (kept in JS)
        const rect = buttonElement.getBoundingClientRect();
        listContainer.style.top = `${window.scrollY + rect.bottom + 5}px`;
        listContainer.style.left = `${window.scrollX + rect.left}px`;

        flamegraphs.forEach(fg => {
            const link = document.createElement('a');
            link.href = fg.path;
            link.textContent = fg.runName;
            link.className = 'download-list-link';
            link.onclick = (e) => {
                e.preventDefault();
                window.open(fg.path, '_blank');
                listContainer.remove();
            };
            listContainer.appendChild(link);
        });

        document.body.appendChild(listContainer);

        setTimeout(() => {
            document.addEventListener('click', function closeHandler(event) {
                if (!listContainer.contains(event.target) && !buttonElement.contains(event.target)) {
                    listContainer.remove();
                    document.removeEventListener('click', closeHandler);
                }
            });
        }, 0);
    }

    // Create Download Chart button (adapts to mode)
    const chartButton = document.createElement('button');
    chartButton.className = 'download-button chart-download-button';
    chartButton.style.marginRight = '8px';

    // Function to update button based on current mode
    function updateChartButton() {
        if (isFlameGraphEnabled()) {
            const flamegraphs = getFlameGraphsForBenchmark(data.label, activeRuns);
            if (flamegraphs.length === 0) {
                chartButton.textContent = 'No Flamegraph Available';
                chartButton.disabled = true;
            } else if (flamegraphs.length === 1) {
                chartButton.textContent = 'Download Flamegraph';
                chartButton.disabled = false;
                chartButton.onclick = (event) => {
                    event.stopPropagation();
                    window.open(flamegraphs[0].path, '_blank');
                };
            } else {
                chartButton.textContent = 'Download Flamegraphs';
                chartButton.disabled = false;
                chartButton.onclick = (event) => {
                    event.stopPropagation();
                    showFlameGraphList(flamegraphs, chartButton);
                };
            }
        } else {
            chartButton.textContent = 'Download Chart';
            chartButton.disabled = false;
            chartButton.onclick = (event) => {
                event.stopPropagation();
                downloadChart(canvasId, data.label);
            };
        }
    }

    updateChartButton();

    // Store the update function on the button so it can be called when mode changes
    chartButton.updateChartButton = updateChartButton;

    // Append the chart download button to the summary
    summary.appendChild(chartButton);
    details.appendChild(summary);

    // Create and append extra info
    const extraInfo = document.createElement('div');
    extraInfo.className = 'extra-info';
    extraInfo.innerHTML = generateExtraInfo(data, 'benchmark');
    details.appendChild(extraInfo);

    footerSection.appendChild(details);
    container.appendChild(footerSection);

    return container;
}

function metadataForLabel(label, type) {
    // First try exact match
    if (benchmarkMetadata[label]?.type === type) {
        return benchmarkMetadata[label];
    }

    // Then fall back to prefix match for backward compatibility
    for (const [key, metadata] of Object.entries(benchmarkMetadata)) {
        if (metadata.type === type && label.startsWith(key)) {
            return metadata;
        }
    }
    return null;
}

// Pre-compute a lookup for the latest run per label
function createLatestRunsLookup() {
    const latestRunsMap = new Map();

    loadedBenchmarkRuns.forEach(run => {
        const runDate = run.date;
        run.results.forEach(result => {
            const label = result.label;
            if (!latestRunsMap.has(label) || runDate > latestRunsMap.get(label).date) {
                latestRunsMap.set(label, {
                    run,
                    result
                });
            }
        });
    });

    return latestRunsMap;
}

function extractLabels(data) {
    // For layer comparison charts
    if (data.benchmarkLabels) {
        return data.benchmarkLabels;
    }

    // For bar charts
    if (data.datasets) {
        // Use the unique lookupLabel for filtering and lookup purposes
        return data.datasets.map(dataset => dataset.lookupLabel || dataset.label);
    }

    // For time series charts
    return [data.label];
}

function getDisplayLabel(label, data, metadata) {
    if (data.datasets) {
        // For bar charts, find the corresponding dataset and use its display label
        const dataset = data.datasets.find(d => (d.lookupLabel || d.label) === label);
        if (dataset) {
            return dataset.label;
        }
    } else if (metadata && metadata.display_name) {
        // For other chart types
        return metadata.display_name;
    }
    return label;
}

function generateExtraInfo(data, type = 'benchmark') {
    const labels = extractLabels(data);

    return labels.map(label => {
        const metadata = metadataForLabel(label, type);
        const latestRun = latestRunsLookup.get(label);
        const displayLabel = getDisplayLabel(label, data, metadata);

        let html = '<div class="extra-info-entry">';

        if (metadata && latestRun) {
            html += `<strong>${displayLabel}:</strong> ${formatCommand(latestRun.result)}<br>`;

            if (metadata.description) {
                html += `<em>Description:</em> ${metadata.description}`;
            }

            if (metadata.notes) {
                html += `<br><em>Notes:</em> <span class="note-text">${metadata.notes}</span>`;
            }

            if (metadata.unstable) {
                html += `<br><em class="unstable-warning">⚠️ Unstable:</em> <span class="unstable-text">${metadata.unstable}</span>`;
            }
        } else {
            html += `<strong>${displayLabel}:</strong> No data available`;
        }

        html += '</div>';
        return html;
    }).join('');
}

function formatCommand(run) {
    const envVars = Object.entries(run.env || {}).map(([key, value]) => `${key}=${value}`).join(' ');
    let command = run.command ? [...run.command] : [];

    return `${envVars} ${command.join(' ')}`.trim();
}

function downloadChart(canvasId, label) {
    const chart = chartInstances.get(canvasId);
    if (chart) {
        const link = document.createElement('a');
        link.href = chart.toBase64Image('image/png', 1)
        link.download = `${label}.png`;
        link.click();
    }
}

function downloadFlameGraph(benchmarkLabel, activeRuns, buttonElement) {
    const flamegraphsToShow = getFlameGraphsForBenchmark(benchmarkLabel, activeRuns);

    if (flamegraphsToShow.length === 0) {
        alert('No flamegraph data available for download');
        return;
    }

    // If there's only one flamegraph, download it directly.
    if (flamegraphsToShow.length === 1) {
        const link = document.createElement('a');
        link.href = flamegraphsToShow[0].path;
        link.download = `${flamegraphsToShow[0].runName}_${benchmarkLabel}.svg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        return;
    }

    // --- Floating list for multiple flamegraphs ---

    // Remove any existing lists first
    const oldList = document.querySelector('.flamegraph-download-list');
    if (oldList) {
        oldList.remove();
    }

    const listContainer = document.createElement('div');
    listContainer.className = 'flamegraph-download-list';

    // Position the list relative to the button
    const rect = buttonElement.getBoundingClientRect();
    listContainer.style.position = 'absolute';
    listContainer.style.top = `${window.scrollY + rect.bottom}px`;
    listContainer.style.left = `${window.scrollX + rect.left}px`;

    flamegraphsToShow.forEach(flamegraph => {
        const link = document.createElement('a');
        link.href = flamegraph.path;
        const filename = `${flamegraph.runName}_${benchmarkLabel}.svg`;
        link.textContent = filename;
        link.download = filename;

        // When a file is clicked, remove the list
        link.onclick = () => {
            listContainer.remove();
        };

        listContainer.appendChild(link);
    });

    // Add a listener to close the list if user clicks elsewhere
    setTimeout(() => {
        document.addEventListener('click', function closeHandler(event) {
            if (!listContainer.contains(event.target)) {
                listContainer.remove();
                document.removeEventListener('click', closeHandler);
            }
        });
    }, 0);

    document.body.appendChild(listContainer);
}

// URL and filtering functions
//
// Information about currently displayed charts, filters, etc. are preserved in
// the URL query string: This allows users to save/share links reproducing exact
// queries, filters, settings, etc. Therefore, for consistency, the URL needs to
// be reconstruted everytime queries, filters, etc. are changed.

function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

function updateURL() {
    const url = new URL(window.location);
    const regex = document.getElementById('bench-filter').value;
    const activeSuites = getActiveSuites();
    const activeRunsList = Array.from(activeRuns);
    const activeTagsList = Array.from(activeTags);

    if (regex) {
        url.searchParams.set('regex', regex);
    } else {
        url.searchParams.delete('regex');
    }

    // Include suites parameter if not all suites are selected
    // OR if the original URL had a suites parameter (preserve user's explicit selection)
    const hadSuitesParam = getQueryParam('suites') !== null;
    if (activeSuites.length > 0 && (activeSuites.length != suiteNames.size || hadSuitesParam)) {
        url.searchParams.set('suites', activeSuites.join(','));
    } else {
        url.searchParams.delete('suites');
    }

    // Add tags to URL
    if (activeTagsList.length > 0) {
        url.searchParams.set('tags', activeTagsList.join(','));
    } else {
        url.searchParams.delete('tags');
    }

    // Handle the runs parameter
    if (activeRunsList.length > 0) {
        // Check if the active runs are the same as default runs
        const defaultRuns = new Set(defaultCompareNames || []);
        const isDefaultRuns = activeRunsList.length === defaultRuns.size &&
            activeRunsList.every(run => defaultRuns.has(run));

        if (isDefaultRuns) {
            // If it's just the default runs, omit the parameter entirely
            url.searchParams.delete('runs');
        } else {
            url.searchParams.set('runs', activeRunsList.join(','));
        }
    } else {
        url.searchParams.delete('runs');
    }

    // Update toggle states in URL using the generic helper
    Object.entries(toggleConfigs).forEach(([toggleId, config]) => {
        updateToggleURL(toggleId, config, url);
    });

    if (!isFlameGraphEnabled()) {
        url.searchParams.delete('flamegraph');
    } else {
        url.searchParams.set('flamegraph', 'true');
    }

    history.replaceState(null, '', url);
}

function filterCharts() {
    const regexInput = document.getElementById('bench-filter').value;
    const regex = new RegExp(regexInput, 'i');
    const activeSuites = getActiveSuites();

    document.querySelectorAll('.chart-container').forEach(container => {
        const label = container.getAttribute('data-label');
        const suite = container.getAttribute('data-suite');
        const isUnstable = container.getAttribute('data-unstable') === 'true';
        const tags = container.getAttribute('data-tags') ?
            container.getAttribute('data-tags').split(',') : [];

        // Check if benchmark has all active tags (if any are selected)
        const hasAllActiveTags = activeTags.size === 0 ||
            Array.from(activeTags).every(tag => tags.includes(tag));

        // Hide unstable benchmarks if showUnstable is false
        const shouldShow = regex.test(label) &&
            activeSuites.includes(suite) &&
            (isUnstableEnabled() || !isUnstable) &&
            hasAllActiveTags;

        container.classList.toggle('hidden', !shouldShow);
    });

    // Don't update URL during initial page load to preserve URL parameters
    if (!isInitializing) {
        updateURL();
    }
}

function getActiveSuites() {
    return Array.from(document.querySelectorAll('.suite-checkbox:checked'))
        .map(checkbox => checkbox.getAttribute('data-suite'));
}

// Data processing
function processTimeseriesData() {
    const resultsByLabel = {};

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            const metadata = metadataForLabel(result.label, 'benchmark');

            if (!resultsByLabel[result.label]) {
                resultsByLabel[result.label] = {
                    label: result.label,
                    display_label: metadata?.display_name || result.label,
                    suite: result.suite,
                    unit: result.unit,
                    lower_is_better: result.lower_is_better,
                    range_min: metadata?.range_min ?? null, // can't use || because js treats 0 as null
                    range_max: metadata?.range_max ?? null,
                    runs: {}
                };
            }
            addRunDataPoint(resultsByLabel[result.label], run, result, false, run.name);
        });
    });

    return Object.values(resultsByLabel);
}

function processBarChartsData() {
    const groupedResults = {};

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            const resultMetadata = metadataForLabel(result.label, 'benchmark');
            const explicitGroup = resultMetadata?.explicit_group || result?.explicit_group;
            if (!explicitGroup) return;

            if (!groupedResults[explicitGroup]) {
                // Look up group metadata
                const groupMetadata = metadataForLabel(explicitGroup, 'group');

                groupedResults[explicitGroup] = {
                    label: explicitGroup,
                    display_label: groupMetadata?.display_name || explicitGroup, // Use display_name if available
                    suite: result.suite,
                    unit: result.unit,
                    lower_is_better: result.lower_is_better,
                    range_min: groupMetadata?.range_min ?? null, // can't use || because js treats 0 as null
                    range_max: groupMetadata?.range_max ?? null,
                    labels: [],
                    datasets: [],
                    // Add metadata if available
                    description: groupMetadata?.description || null,
                    notes: groupMetadata?.notes || null,
                    unstable: groupMetadata?.unstable || null
                };
            }

            const group = groupedResults[explicitGroup];

            if (!group.labels.includes(run.name)) {
                group.labels.push(run.name);
            }

            // Store the label we'll use for lookup and the display label separately
            const lookupLabel = result.label;
            // First try to get display name from metadata using the actual label
            const metadata = benchmarkMetadata[result.label];
            const displayLabel = metadata?.display_name || result.label;

            let dataset = group.datasets.find(d => d.lookupLabel === lookupLabel);
            if (!dataset) {
                const datasetIndex = group.datasets.length;
                dataset = {
                    lookupLabel: lookupLabel, // Store the original label for lookup
                    label: displayLabel,      // Use display label for rendering
                    data: new Array(group.labels.length).fill(null),
                    backgroundColor: colorPalette[datasetIndex % colorPalette.length],
                    borderColor: colorPalette[datasetIndex % colorPalette.length],
                    borderWidth: 1
                };
                group.datasets.push(dataset);
            }

            const runIndex = group.labels.indexOf(run.name);
            if (dataset.data[runIndex] == null)
                dataset.data[runIndex] = result.value;
        });
    });

    return Object.values(groupedResults);
}

function getLayerTags(metadata) {
    const layerTags = new Set();
    if (metadata?.tags) {
        metadata.tags.forEach(tag => {
            if (tag.startsWith('SYCL') || tag.startsWith('UR') || tag === 'L0') {
                layerTags.add(tag);
            }
        });
    }
    return layerTags;
}

function processLayerComparisonsData() {
    const groupedResults = {};
    const labelsByGroup = {};

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            const resultMetadata = metadataForLabel(result.label, 'benchmark');
            const explicitGroup = resultMetadata?.explicit_group || result.explicit_group;
            if (!explicitGroup) return;

            if (!labelsByGroup[explicitGroup]) {
                labelsByGroup[explicitGroup] = new Set();
            }
            labelsByGroup[explicitGroup].add(result.label);
        });
    });

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            // Get explicit_group from metadata
            const resultMetadata = metadataForLabel(result.label, 'benchmark');
            const explicitGroup = resultMetadata?.explicit_group || result.explicit_group;
            if (!explicitGroup) return;

            // Skip if no metadata available
            const metadata = metadataForLabel(explicitGroup, 'group');
            if (!metadata) return;

            // Get all benchmark labels in this group
            const labelsInGroup = labelsByGroup[explicitGroup];

            // Check if this group compares different layers
            const uniqueLayers = new Set();
            labelsInGroup.forEach(label => {
                const labelMetadata = metadataForLabel(label, 'benchmark');
                const layerTags = getLayerTags(labelMetadata);
                layerTags.forEach(tag => uniqueLayers.add(tag));
            });

            // Only process groups that compare different layers
            if (uniqueLayers.size <= 1) return;

            if (!groupedResults[explicitGroup]) {
                groupedResults[explicitGroup] = {
                    label: explicitGroup,
                    suite: result.suite,
                    unit: result.unit,
                    lower_is_better: result.lower_is_better,
                    range_min: metadata?.range_min ?? null, // can't use || because js treats 0 as null
                    range_max: metadata?.range_max ?? null,
                    runs: {},
                    benchmarkLabels: [],
                    description: metadata?.description || null,
                    notes: metadata?.notes || null,
                    unstable: metadata?.unstable || null
                };
            }

            const group = groupedResults[explicitGroup];
            const name = result.label + ' (' + run.name + ')';

            // Add the benchmark label if it's not already in the array
            if (!group.benchmarkLabels.includes(result.label)) {
                group.benchmarkLabels.push(result.label);
            }

            addRunDataPoint(group, run, result, true, name);
        });
    });

    return Object.values(groupedResults);
}

function addRunDataPoint(group, run, result, comparison, name = null) {
    const runKey = name || result.label + ' (' + run.name + ')';

    if (!group.runs[runKey]) {
        const datasetIndex = Object.keys(group.runs).length;
        const metadata = benchmarkMetadata[result.name];
        const displayName = metadata?.display_name || result.label;
        group.runs[runKey] = {
            label: runKey,
            displayLabel: displayName + ' (' + run.name + ')', // Format for layer comparison charts
            runName: run.name,
            data: [],
            borderColor:
                comparison ? colorPalette[datasetIndex % colorPalette.length] : getColorForName(run.name),
            backgroundColor:
                comparison ? colorPalette[datasetIndex % colorPalette.length] : getColorForName(run.name),
            borderWidth: 1,
            pointRadius: 3,
            pointStyle: 'circle',
            pointHoverRadius: 5
        };
    }

    group.runs[runKey].data.push({
        // For historical results use only run.name, for layer comparisons use displayLabel
        seriesName: name === run.name ? run.name : group.runs[runKey].displayLabel,
        x: new Date(run.date),
        y: result.value,
        stddev: result.stddev,
        gitHash: run.git_hash,
        gitRepo: run.github_repo,
        compute_runtime: run.compute_runtime,
        gitBenchUrl: result.git_url,
        gitBenchHash: result.git_hash,
    });

    return group;
}

// Setup functions
function setupRunSelector() {
    runSelect = document.getElementById('run-select');
    selectedRunsDiv = document.getElementById('selected-runs');

    // Clear existing options first to prevent duplicates when reloading with archived data
    runSelect.innerHTML = '';

    allRunNames.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        runSelect.appendChild(option);
    });

    updateSelectedRuns(false);
}

function setupSuiteFilters() {
    suiteFiltersContainer = document.getElementById('suite-filters');

    // Clear existing suite filters before adding new ones
    suiteFiltersContainer.innerHTML = '';

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            suiteNames.add(result.suite);
        });
    });

    // Debug logging for suite names
    console.log('Available suites:', Array.from(suiteNames));
    console.log('Loaded benchmark runs:', loadedBenchmarkRuns.map(run => ({
        name: run.name,
        suites: [...new Set(run.results.map(r => r.suite))]
    })));

    suiteNames.forEach(suite => {
        const label = document.createElement('label');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'suite-checkbox';
        checkbox.dataset.suite = suite;
        checkbox.checked = true;
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(' ' + suite));
        suiteFiltersContainer.appendChild(label);
        suiteFiltersContainer.appendChild(document.createTextNode(' '));
    });
}

function isNotesEnabled() {
    return isToggleEnabled('show-notes');
}

function isUnstableEnabled() {
    return isToggleEnabled('show-unstable');
}

function isCustomRangesEnabled() {
    return isToggleEnabled('custom-range');
}

function isArchivedDataEnabled() {
    return isToggleEnabled('show-archived-data');
}

function isFlameGraphEnabled() {
    const flameGraphToggle = document.getElementById('show-flamegraph');
    return flameGraphToggle ? flameGraphToggle.checked : false;
}

function validateFlameGraphData() {
    return window.flamegraphData?.runs !== undefined;
}

function sanitizeFilename(name) {
    /**
     * Sanitize a string to be safe for use as a filename or directory name.
     * Replace invalid characters (including space) with underscores so paths are shell-safe.
     *
     * Invalid characters: " : < > | * ? \r \n <space>
     */
    const invalidChars = /[":;<>|*?\r\n ]/g; // Added space to align with Python implementation
    return name.replace(invalidChars, '_');
}

function createFlameGraphPath(benchmarkLabel, runName, timestamp) {
    const suiteName = window.flamegraphData?.runs?.[runName]?.suites?.[benchmarkLabel];

    if (!suiteName) {
        console.error(`Could not find suite for benchmark '${benchmarkLabel}' in run '${runName}'`);
        // Fallback: sanitize benchmark name for directory structure
        const sanitizedBenchmarkName = sanitizeFilename(benchmarkLabel);
        const timestampPrefix = timestamp + '_';
        const relativePath = `results/flamegraphs/${sanitizedBenchmarkName}/${timestampPrefix}${runName}.svg`;

        // For local mode, use relative path; for remote mode, use full URL
        const baseUrl = getResourceBaseUrl();
        return baseUrl === '.' ? relativePath : `${baseUrl}/${relativePath}`;
    }

    // Apply sanitization to both suite and benchmark names to match Python implementation
    const sanitizedSuiteName = sanitizeFilename(suiteName);
    const sanitizedBenchmarkName = sanitizeFilename(benchmarkLabel);
    const benchmarkDirName = `${sanitizedSuiteName}__${sanitizedBenchmarkName}`;
    const timestampPrefix = timestamp + '_';
    const relativePath = `results/flamegraphs/${benchmarkDirName}/${timestampPrefix}${runName}.svg`;

    // For local mode, use relative path; for remote mode, use full URL
    const baseUrl = getResourceBaseUrl();
    return baseUrl === '.' ? relativePath : `${baseUrl}/${relativePath}`;
}

function getRunsWithFlameGraph(benchmarkLabel, activeRuns) {
    // Inline validation for better performance
    if (!window.flamegraphData?.runs) {
        return [];
    }

    const runsWithFlameGraph = [];
    activeRuns.forEach(runName => {
        if (window.flamegraphData.runs[runName] &&
            window.flamegraphData.runs[runName].suites &&
            Object.keys(window.flamegraphData.runs[runName].suites).includes(benchmarkLabel)) {
            runsWithFlameGraph.push({
                name: runName,
                timestamp: window.flamegraphData.runs[runName].timestamp
            });
        }
    });

    return runsWithFlameGraph;
}

function getFlameGraphsForBenchmark(benchmarkLabel, activeRuns) {
    const runsWithFlameGraph = getRunsWithFlameGraph(benchmarkLabel, activeRuns);
    const flamegraphsToShow = [];

    // For each run that has flamegraph data, create the path
    runsWithFlameGraph.forEach(runInfo => {
        const flamegraphPath = createFlameGraphPath(benchmarkLabel, runInfo.name, runInfo.timestamp);

        flamegraphsToShow.push({
            path: flamegraphPath,
            runName: runInfo.name,
            timestamp: runInfo.timestamp
        });
    });

    // Sort by the order of activeRuns to maintain consistent display order
    const runOrder = Array.from(activeRuns);
    flamegraphsToShow.sort((a, b) => {
        const indexA = runOrder.indexOf(a.runName);
        const indexB = runOrder.indexOf(b.runName);
        return indexA - indexB;
    });

    return flamegraphsToShow;
}

function updateFlameGraphTooltip() {
    const flameGraphToggle = document.getElementById('show-flamegraph');
    const label = document.querySelector('label[for="show-flamegraph"]');

    if (!flameGraphToggle || !label) return;

    // Check if we have flamegraph data
    if (validateFlameGraphData()) {
        const runsWithFlameGraphs = Object.keys(window.flamegraphData.runs).filter(
            runName => window.flamegraphData.runs[runName].suites &&
                Object.keys(window.flamegraphData.runs[runName].suites).length > 0
        );

        if (runsWithFlameGraphs.length > 0) {
            label.title = `Show flamegraph SVG files instead of benchmark charts. Available for runs: ${runsWithFlameGraphs.join(', ')}`;
            flameGraphToggle.disabled = false;
            label.classList.remove('disabled-text');
        } else {
            label.title = 'No flamegraph data available - run benchmarks with --flamegraph option to enable';
            flameGraphToggle.disabled = true;
            label.classList.add('disabled-text');
        }
    } else {
        label.title = 'No flamegraph data available - run benchmarks with --flamegraph option to enable';
        flameGraphToggle.disabled = true;
        label.classList.add('disabled-text');
    }
}

function setupToggles() {
    // Set up configured generic toggles
    Object.entries(toggleConfigs).forEach(([toggleId, config]) => {
        setupToggle(toggleId, config);
    });

    // No additional per-toggle setup required; 'show-flamegraph' is handled
    // by the generic `toggleConfigs` entry and its onChange handler.
}

function setupTagFilters() {
    tagFiltersContainer = document.getElementById('tag-filters');

    // Clear existing tag filters before adding new ones
    tagFiltersContainer.innerHTML = '';

    const allTags = [];

    if (benchmarkTags) {
        for (const tag in benchmarkTags) {
            if (!allTags.includes(tag)) {
                allTags.push(tag);
            }
        }
    }

    // Create tag filter elements
    allTags.forEach(tag => {
        const tagContainer = document.createElement('div');
        tagContainer.className = 'tag-filter';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `tag-${tag}`;
        checkbox.className = 'tag-checkbox';
        checkbox.dataset.tag = tag;

        const label = document.createElement('label');
        label.htmlFor = `tag-${tag}`;
        label.textContent = tag;

        // Add info icon with tooltip if tag description exists
        if (benchmarkTags[tag]) {
            const infoIcon = document.createElement('span');
            infoIcon.className = 'tag-info';
            infoIcon.textContent = 'ⓘ';
            infoIcon.title = benchmarkTags[tag].description;
            label.appendChild(infoIcon);
        }

        checkbox.addEventListener('change', function () {
            if (this.checked) {
                activeTags.add(tag);
            } else {
                activeTags.delete(tag);
            }
            filterCharts();
        });

        tagContainer.appendChild(checkbox);
        tagContainer.appendChild(label);
        tagFiltersContainer.appendChild(tagContainer);
    });
}

function toggleAllTags(select) {
    const checkboxes = document.querySelectorAll('.tag-checkbox');

    checkboxes.forEach(checkbox => {
        checkbox.checked = select;
        const tag = checkbox.dataset.tag;

        if (select) {
            activeTags.add(tag);
        } else {
            activeTags.delete(tag);
        }
    });

    filterCharts();
}

function initializeCharts() {
    console.log('initializeCharts() started');
    console.log('loadedBenchmarkRuns:', loadedBenchmarkRuns.length, 'runs');
    console.log('First run name:', loadedBenchmarkRuns.length > 0 ? loadedBenchmarkRuns[0].name : 'no runs');
    console.log('defaultCompareNames:', defaultCompareNames);

    // Process raw data
    console.log('Processing timeseries data...');
    timeseriesData = processTimeseriesData();
    console.log('Timeseries data processed:', timeseriesData.length, 'items');

    console.log('Processing bar charts data...');
    barChartsData = processBarChartsData();
    console.log('Bar charts data processed:', barChartsData.length, 'items');

    console.log('Processing layer comparisons data...');
    layerComparisonsData = processLayerComparisonsData();
    console.log('Layer comparisons data processed:', layerComparisonsData.length, 'items');

    allRunNames = [...new Set(loadedBenchmarkRuns.map(run => run.name))];
    console.log('All run names:', allRunNames);

    // In flamegraph-only mode, ensure we include runs from flamegraph data
    if (validateFlameGraphData()) {
        const flamegraphRunNames = Object.keys(window.flamegraphData.runs);
        allRunNames = [...new Set([...allRunNames, ...flamegraphRunNames])];
        console.log('Added flamegraph runs, total run names:', allRunNames);
    }

    latestRunsLookup = createLatestRunsLookup();
    console.log('Run names and lookup created. Runs:', allRunNames);

    // Check if we have actual benchmark results vs flamegraph-only results
    const hasActualBenchmarks = loadedBenchmarkRuns.some(run =>
        run.results && run.results.length > 0 && run.results.some(result => result.suite !== 'flamegraph')
    );

    const hasFlameGraphResults = loadedBenchmarkRuns.some(run =>
        run.results && run.results.some(result => result.suite === 'flamegraph')
    ) || (validateFlameGraphData() && Object.keys(window.flamegraphData.runs).length > 0);

    console.log('Benchmark analysis:', {
        hasActualBenchmarks,
        hasFlameGraphResults,
        loadedBenchmarkRuns: loadedBenchmarkRuns.length,
        runDetails: loadedBenchmarkRuns.map(run => ({
            name: run.name,
            resultCount: run.results ? run.results.length : 0,
            hasResults: run.results && run.results.length > 0
        })),
        flamegraphValidation: validateFlameGraphData(),
        flamegraphRunCount: validateFlameGraphData() ? Object.keys(window.flamegraphData.runs).length : 0
    });

    // If we only have flamegraph results (no actual benchmark data), create synthetic data
    if (!hasActualBenchmarks && hasFlameGraphResults) {
        console.log('Detected flamegraph-only mode - creating synthetic data for flamegraphs');

        // Check if we have flamegraph data available
        const hasFlamegraphData = validateFlameGraphData() &&
            Object.keys(window.flamegraphData.runs).length > 0 &&
            Object.values(window.flamegraphData.runs).some(run => run.suites && Object.keys(run.suites).length > 0);

        if (hasFlamegraphData) {
            console.log('Creating synthetic benchmark data for flamegraph display');
            createFlameGraphOnlyData();

            // Auto-enable flamegraph mode for user convenience
            const flameGraphToggle = document.getElementById('show-flamegraph');
            if (flameGraphToggle && !flameGraphToggle.checked) {
                flameGraphToggle.checked = true;
                console.log('Auto-enabled flamegraph view for flamegraph-only data');
            }
        } else {
            console.log('No flamegraph data available - showing message');
            displayNoDataMessage();
        }
    } else if (!hasActualBenchmarks && !hasFlameGraphResults) {
        // No runs and no results - something went wrong
        console.log('No benchmark data found at all');
        displayNoDataMessage();
    }

    // Create global options map for annotations
    annotationsOptions = createAnnotationsOptions();
    // Make it available to the ChartAnnotations module
    window.annotationsOptions = annotationsOptions;

    // Set up active runs
    const runsParam = getQueryParam('runs');
    if (runsParam) {
        const runsFromUrl = runsParam.split(',');

        // Start with an empty set
        activeRuns = new Set();

        // Process each run from URL
        runsFromUrl.forEach(run => {
            if (run === 'default') {
                // Special case: include all default runs
                (defaultCompareNames || []).forEach(defaultRun => {
                    if (allRunNames.includes(defaultRun)) {
                        activeRuns.add(defaultRun);
                    }
                });
            } else if (allRunNames.includes(run)) {
                // Add the specific run if it exists
                activeRuns.add(run);
            }
        });
    } else {
        // No runs parameter, use defaults
        activeRuns = new Set(defaultCompareNames || []);

        // If no default runs and we're in flamegraph-only mode, use all available runs
        if (activeRuns.size === 0 && !hasActualBenchmarks && hasFlameGraphResults) {
            activeRuns = new Set(allRunNames);
            console.log('Flamegraph-only mode: auto-selected all available runs:', Array.from(activeRuns));
        }
    }

    // Setup UI components
    setupRunSelector();
    setupSuiteFilters();
    setupToggles();
    initializePlatformTab();
    // Setup tag filters after everything else is ready
    setupTagFilters();

    // Apply URL parameters
    const regexParam = getQueryParam('regex');
    const suitesParam = getQueryParam('suites');
    const tagsParam = getQueryParam('tags');

    if (regexParam) {
        document.getElementById('bench-filter').value = regexParam;
    }

    if (suitesParam) {
        const suites = suitesParam.split(',');
        document.querySelectorAll('.suite-checkbox').forEach(checkbox => {
            checkbox.checked = suites.includes(checkbox.getAttribute('data-suite'));
        });
    }

    // Apply tag filters from URL
    if (tagsParam) {
        const tags = tagsParam.split(',');
        tags.forEach(tag => {
            const checkbox = document.querySelector(`.tag-checkbox[data-tag="${tag}"]`);
            if (checkbox) {
                checkbox.checked = true;
                activeTags.add(tag);
            }
        });
    }

    // Setup event listeners
    document.querySelectorAll('.suite-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', filterCharts);
    });
    document.getElementById('bench-filter').addEventListener('input', filterCharts);

    // Draw initial charts
    updateCharts();

    // Mark initialization as complete - URL parameters have been applied
    isInitializing = false;

    // Update the URL to ensure it reflects the final state after initialization
    updateURL();
}

// Make functions available globally for onclick handlers
window.addSelectedRun = addSelectedRun;
window.removeRun = removeRun;
window.toggleAllTags = toggleAllTags;

// Helper function to fetch and process benchmark data
function fetchAndProcessData(url, isArchived = false) {
    const loadingIndicator = document.getElementById('loading-indicator');
    return fetch(url)
        .then(resp => { if (!resp.ok) throw new Error(`Got response status ${resp.status}.`); return resp.json(); })
        .then(data => {
            const runsArray = Array.isArray(data.benchmarkRuns) ? data.benchmarkRuns : data.runs;
            if (!runsArray || !Array.isArray(runsArray)) {
                throw new Error('Invalid data format: expected benchmarkRuns or runs array');
            }
            if (isArchived) {
                loadedBenchmarkRuns = loadedBenchmarkRuns.concat(runsArray);
                archivedDataLoaded = true;
            } else {
                loadedBenchmarkRuns = runsArray;
                window.benchmarkMetadata = data.benchmarkMetadata || data.metadata || {};
                window.benchmarkTags = data.benchmarkTags || data.tags || {};
                window.flamegraphData = (data.flamegraphData && data.flamegraphData.runs) ? data.flamegraphData : { runs: {} };
                if (Array.isArray(data.defaultCompareNames) && (!defaultCompareNames || defaultCompareNames.length === 0)) {
                    defaultCompareNames = data.defaultCompareNames;
                }
                console.log('Remote data loaded (normalized):', {
                    runs: runsArray.length,
                    metadata: Object.keys(window.benchmarkMetadata).length,
                    tags: Object.keys(window.benchmarkTags).length,
                    flamegraphs: Object.keys(window.flamegraphData.runs).length,
                    defaultCompareNames: defaultCompareNames
                });
            }

            // The following variables have same values regardless of whether
            // we load archived or current data
            benchmarkMetadata = data.metadata || benchmarkMetadata || {};
            benchmarkTags = data.tags || benchmarkTags || {};

            initializeCharts();
        })
        .catch(err => {
            console.error(`Error fetching ${isArchived ? 'archived' : 'remote'} data:`, err);
            loadingIndicator.textContent = 'Fetching remote data failed.';
        })
        .finally(() => loadingIndicator.classList.add('hidden'));
}

// Load data based on configuration
function loadData() {
    console.log('=== loadData() called ===');
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.classList.remove('hidden'); // Show loading indicator

    if (typeof remoteDataUrl !== 'undefined' && remoteDataUrl !== '') {
        url = remoteDataUrl.endsWith('/') ? remoteDataUrl + 'data.json' : remoteDataUrl + '/data.json';
        console.log('Using remote data URL:', url);
        // Fetch data from remote URL
        fetchAndProcessData(url);
    } else {
        console.log('Using local canonical data');
        if (!Array.isArray(window.benchmarkRuns)) {
            console.error('benchmarkRuns missing or invalid');
            loadedBenchmarkRuns = [];
            window.benchmarkMetadata = {};
            window.benchmarkTags = {};
            window.flamegraphData = { runs: {} };
        } else {
            loadedBenchmarkRuns = window.benchmarkRuns;
            window.benchmarkMetadata = window.benchmarkMetadata || {};
            window.benchmarkTags = window.benchmarkTags || {};
            window.flamegraphData = (window.flamegraphData && window.flamegraphData.runs) ? window.flamegraphData : { runs: {} };
            console.log('Local data loaded (standalone globals):', {
                runs: loadedBenchmarkRuns.length,
                metadata: Object.keys(window.benchmarkMetadata).length,
                tags: Object.keys(window.benchmarkTags).length,
                flamegraphs: Object.keys(window.flamegraphData.runs).length
            });
        }
        initializeCharts();
        if (loadedBenchmarkRuns.length === 0) {
            loadingIndicator.textContent = 'No benchmark data found.';
            loadingIndicator.setAttribute('role', 'alert');   // optional accessibility
        } else {
            loadingIndicator.classList.add('hidden');         // hide when data present
        }
        console.log('=== loadData() completed ===');
    }
}

// Function to load archived data and merge with current data
// Archived data consists of older benchmark results that have been separated from
// the primary dataset but are still available for historical analysis.
function loadArchivedData() {
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.classList.remove('hidden');

    if (archivedDataLoaded) {
        updateCharts();
        loadingIndicator.classList.add('hidden');
        return;
    }

    if (typeof remoteDataUrl !== 'undefined' && remoteDataUrl !== '') {
        // Fetch data from remote URL
        const url = remoteDataUrl.endsWith('/') ? remoteDataUrl + 'data_archive.json' : remoteDataUrl + '/data_archive.json';
        fetchAndProcessData(url, true);
    } else {
        // For local data use a static js file
        const script = document.createElement('script');
        script.src = 'data_archive.js';
        script.onload = () => {
            // Merge the archived runs with current runs
            loadedBenchmarkRuns = loadedBenchmarkRuns.concat(benchmarkRuns);
            archivedDataLoaded = true;
            initializeCharts();
            loadingIndicator.classList.add('hidden');
        };

        script.onerror = () => {
            console.error('Failed to load data_archive.js');
            loadingIndicator.classList.add('hidden');
        };

        document.head.appendChild(script);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    loadData();
});

// Process all benchmark runs to create a global options map for annotations
function createAnnotationsOptions() {
    const repoMap = new Map();

    loadedBenchmarkRuns.forEach(run => {
        run.results.forEach(result => {
            if (result.git_url && !repoMap.has(result.git_url)) {
                const suiteName = result.suite;
                const colorIndex = repoMap.size % annotationPalette.length;
                const backgroundColor = annotationPalette[colorIndex].replace('0.8', '0.9');
                const color = {
                    border: annotationPalette[colorIndex],
                    background: backgroundColor
                };

                repoMap.set(result.git_url, {
                    name: suiteName,
                    url: result.git_url,
                    color: color,
                });
            }
        });
    });

    return repoMap;
}

function displaySelectedRunsPlatformInfo() {
    const container = document.querySelector('.platform-info .platform');
    if (!container) return;

    container.innerHTML = '';

    // Get platform info for only the selected runs
    const selectedRunsWithPlatform = Array.from(activeRuns)
        .map(runName => {
            const run = loadedBenchmarkRuns.find(r => r.name === runName);
            if (run && run.platform) {
                return { name: runName, platform: run.platform };
            }
            return null;
        })
        .filter(item => item !== null);
    if (selectedRunsWithPlatform.length === 0) {
        container.innerHTML = '<p>No platform information available to display.</p>';
        return;
    }
    selectedRunsWithPlatform.forEach(runData => {
        const runSection = document.createElement('div');
        runSection.className = 'platform-run-section';
        const runTitle = document.createElement('h3');
        runTitle.textContent = `Run: ${runData.name}`;
        runTitle.className = 'platform-run-title';
        runSection.appendChild(runTitle);
        // Create just the platform details without the grid wrapper
        const platform = runData.platform;
        const detailsContainer = document.createElement('div');
        detailsContainer.className = 'platform-details-compact';
        detailsContainer.innerHTML = createPlatformDetailsHTML(platform);
        runSection.appendChild(detailsContainer);
        container.appendChild(runSection);
    });
}

// Platform Information Functions

function createPlatformDetailsHTML(platform) {
    const formattedTimestamp = platform.timestamp ?
        new Date(platform.timestamp).toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }) : 'Unknown';

    return `
        <div class="platform-section compact">
            <div class="platform-item">
                <span class="platform-label">Time:</span>
                <span class="platform-value">${formattedTimestamp}</span>
            </div>
            <div class="platform-item">
                <span class="platform-label">OS:</span>
                <span class="platform-value">${platform.os || 'Unknown'}</span>
            </div>
            <div class="platform-item">
                <span class="platform-label">CPU:</span>
                <span class="platform-value">${platform.cpu_info || 'Unknown'} (${platform.cpu_count || '?'} cores)</span>
            </div>
            <div class="platform-item">
                <span class="platform-label">GPU:</span>
                <div class="platform-value multiple">
                    ${platform.gpu_info && platform.gpu_info.length > 0
            ? platform.gpu_info.map(gpu => `<div class="platform-gpu-item">    • ${gpu}</div>`).join('')
            : '<div class="platform-gpu-item">    • No GPU detected</div>'}
                </div>
            </div>
            <div class="platform-item">
                <span class="platform-label">Driver:</span>
                <span class="platform-value">${platform.gpu_driver_version || 'Unknown'}</span>
            </div>
            <div class="platform-item">
                <span class="platform-label">Tools:</span>
                <span class="platform-value">${platform.gcc_version || '?'} • ${platform.clang_version || '?'} • ${platform.python || '?'}</span>
            </div>
            <div class="platform-item">
                <span class="platform-label">Runtime:</span>
                <span class="platform-value">${platform.level_zero_version || '?'} • compute-runtime ${platform.compute_runtime_version || '?'}</span>
            </div>
        </div>
    `;
}

function initializePlatformTab() {
    displaySelectedRunsPlatformInfo();
}

// Function to create chart data for flamegraph-only mode
function createFlameGraphOnlyData() {
    // Check if we have flamegraphData from data.js
    if (validateFlameGraphData()) {
        // Collect all unique benchmarks from all runs that have flamegraphs
        const allBenchmarks = new Set();
        const availableRuns = Object.keys(window.flamegraphData.runs);

        availableRuns.forEach(runName => {
            if (window.flamegraphData.runs[runName].suites) {
                Object.keys(window.flamegraphData.runs[runName].suites).forEach(benchmark => {
                    allBenchmarks.add(benchmark);
                });
            }
        });

        if (allBenchmarks.size > 0) {
            console.log(`Using flamegraphData from data.js for runs: ${availableRuns.join(', ')}`);
            console.log(`Available benchmarks with flamegraphs: ${Array.from(allBenchmarks).join(', ')}`);
            createSyntheticFlameGraphData(Array.from(allBenchmarks));
            return; // Success - we have flamegraph data
        }
    }

    // No flamegraph data available - benchmarks were run without --flamegraph option
    console.log('No flamegraph data found - benchmarks were likely run without --flamegraph option');

    // Disable the flamegraph checkbox since no flamegraphs are available
    const flameGraphToggle = document.getElementById('show-flamegraph');
    if (flameGraphToggle) {
        flameGraphToggle.disabled = true;
        flameGraphToggle.checked = false;

        // Add a visual indicator that flamegraphs are not available
        const label = document.querySelector('label[for="show-flamegraph"]');
        if (label) {
            label.classList.add('disabled-text');
            label.title = 'No flamegraph data available - run benchmarks with --flamegraph option to enable';
        }

        console.log('Disabled flamegraph toggle - no flamegraph data available');
    }

    // Clear any flamegraph-only mode detection and proceed with normal benchmark display
    // This handles the case where we're in flamegraph-only mode but have no actual flamegraph data
}

function displayNoFlameGraphsMessage() {
    // Clear existing data arrays
    timeseriesData = [];
    barChartsData = [];
    layerComparisonsData = [];

    // Add a special suite for the message
    suiteNames.add('Information');

    // Create a special entry to show a helpful message
    const messageData = {
        label: 'No FlameGraphs Available',
        display_label: 'No FlameGraphs Available',
        suite: 'Information',
        unit: 'message',
        lower_is_better: false,
        range_min: null,
        range_max: null,
        runs: {}
    };

    timeseriesData.push(messageData);
    console.log('Added informational message about missing flamegraphs');
}

function displayNoDataMessage() {
    // Clear existing data arrays
    timeseriesData = [];
    barChartsData = [];
    layerComparisonsData = [];

    // Add a special suite for the message
    suiteNames.add('Information');

    // Create a special entry to show a helpful message
    const messageData = {
        label: 'No Data Available',
        display_label: 'No Benchmark Data Available',
        suite: 'Information',
        unit: 'message',
        lower_is_better: false,
        range_min: null,
        range_max: null,
        runs: {}
    };

    timeseriesData.push(messageData);
    console.log('Added informational message about missing benchmark data');
}

function createSyntheticFlameGraphData(flamegraphLabels) {
    // Clear existing data arrays since we're in flamegraph-only mode
    timeseriesData = [];
    barChartsData = [];
    layerComparisonsData = [];

    // Create synthetic benchmark results for each flamegraph
    flamegraphLabels.forEach(label => {
        // Get suite from flamegraphData - this should always be available
        let suite = null;

        if (window.flamegraphData?.runs) {
            // Check all runs for suite information for this benchmark
            for (const runName in window.flamegraphData.runs) {
                const runData = window.flamegraphData.runs[runName];
                if (runData.suites && runData.suites[label]) {
                    suite = runData.suites[label];
                    break;
                }
            }
        }

        // If no suite found, this indicates a problem with the flamegraph data generation
        if (!suite) {
            console.error(`No suite information found for flamegraph: ${label}. This indicates missing suite data in flamegraphs.js`);
            suite = `ERROR: Missing suite for ${label}`;
        }

        // Add to suite names
        suiteNames.add(suite);

        // Create a synthetic timeseries entry for this flamegraph
        const syntheticData = {
            label: label,
            display_label: label, // Use label directly since this is synthetic data
            suite: suite,
            unit: 'flamegraph',
            lower_is_better: false,
            range_min: null,
            range_max: null,
            runs: {}
        };

        // Add this to timeseriesData so it shows up in the charts
        timeseriesData.push(syntheticData);
    });

    console.log(`Created synthetic data for ${flamegraphLabels.length} flamegraphs with suites:`, Array.from(suiteNames));
}

