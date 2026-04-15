// Copyright (C) 2024-2026 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * Dashboard checkbox toggles: URL-backed state and onChange wiring.
 * Depends on `getQueryParam` from scripts.js (load order: this file before scripts.js;
 * `setupToggle` runs later, after `getQueryParam` is defined).
 */

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
    },
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
    return isToggleEnabled('show-flamegraph');
}

function setupToggles() {
    // Set up configured generic toggles
    Object.entries(toggleConfigs).forEach(([toggleId, config]) => {
        setupToggle(toggleId, config);
    });

    // No additional per-toggle setup required; 'show-flamegraph' is handled
    // by the generic `toggleConfigs` entry and its onChange handler.
}
