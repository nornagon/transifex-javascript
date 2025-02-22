const _ = require('lodash');
const ngHtmlParser = require('angular-html-parser');
const fs = require('fs');
const { mergePayload } = require('../merge');
const { createPayload, isPayloadValid } = require('./utils');

/**
 * Global regexp to find use of TranslatePipe.
 */
const pipeRegexpG = /{{\s*?['|"]([\s\S]+?)['|"]\s*?\|\s*?translate\s*?:?({[\s\S]*?})?\s*?}}/gi;

/**
  * Regexp to find use of TranslatePipe and match with capture groups.
  */
const pipeRegexp = /{{\s*?['|"]([\s\S]+?)['|"]\s*?\|\s*?translate\s*?:?({[\s\S]*?})?\s*?}}/i;

/**
  * Regexp to find use of TranslatePipe in Attributes;
  */
const pipeBindingRegexp = /'([\s\S]+?)'\s*?\|\s*?translate\s*?:?({[\s\S]*?})?/i;

/**
 * Loosely parses string (from HTML) to an object.
 *
 * According to Mozilla a bit better than eval().
 *
 * @param {str} obj
 * @returns {*}
 */
function looseJsonParse(obj) {
  let parsed;

  try {
    // eslint-disable-next-line no-new-func
    parsed = Function(`"use strict";return (${obj})`)();
  } catch (err) {
    parsed = {};
  }

  return parsed;
}

/**
 * Parse an HTML file and detects T/UT tags and usage of TranslatePipe
 *
 * @param {Object} HASHES
 * @param {String} filename
 * @param {String} relativeFile
 * @param {String[]} appendTags
 * @param {Object} options
 * @returns void
 */
function parseHTMLTemplateFile(HASHES, filename, relativeFile, options) {
  const TXComponents = [];
  const TXTemplateStrings = [];

  function parseTemplateText(text) {
    const textStr = _.trim(String(text));

    if (textStr.length) {
      const results = String(textStr).match(pipeRegexpG);

      if (results && results.length) {
        _.each(results, (result) => {
          const lineResult = result.match(pipeRegexp);

          if (lineResult) {
            const string = lineResult[1];
            const paramsStr = lineResult[2];

            const params = looseJsonParse(paramsStr) || {};

            if (string && params) {
              TXTemplateStrings.push({
                string,
                params,
              });
            }
          }
        });
      }
    }
  }

  function parseTemplateBindingText(text) {
    const textStr = _.trim(String(text));

    if (textStr.length) {
      const result = textStr.match(pipeBindingRegexp);

      if (result) {
        const string = result[1];
        const paramsStr = result[2];

        const params = looseJsonParse(paramsStr) || {};

        if (string && params) {
          TXTemplateStrings.push({
            string,
            params,
          });
        }
      }
    }
  }

  function parseTemplateNode(children) {
    if (children) {
      _.each(children, (child) => {
        if (child.name === 'T' || child.name === 'UT') {
          TXComponents.push(child);
        } else if (child.type === 'text') {
          parseTemplateText(child.value);
        } else if (child.attrs && child.attrs.length > 0) {
          const attributes = child.attrs.filter((a) => a.value.includes('translate'));

          _.each(attributes, (attr) => {
            parseTemplateBindingText(attr.value);
          });
        }

        parseTemplateNode(child.children);
      });
    }
  }

  const data = fs.readFileSync(filename, 'utf8');
  const { rootNodes, errors } = ngHtmlParser.parse(data);
  if (errors.length) return;

  parseTemplateNode(rootNodes);
  _.each(TXComponents, (txcmp) => {
    let string = '';
    let key = '';
    const params = {};
    if (txcmp.attrs) {
      _.each(txcmp.attrs, (attribute) => {
        if (attribute.name === 'str') {
          string = attribute.value;
        } else if (attribute.name === 'key') {
          key = attribute.value;
        } else {
          params[attribute.name] = attribute.value;
        }
      });
    }
    if (string) {
      const partial = createPayload(string, params, relativeFile, options);
      if (!isPayloadValid(partial, options)) return;

      mergePayload(HASHES, {
        [key || partial.key]: {
          string: partial.string,
          meta: partial.meta,
        },
      });
    }
  });

  _.each(TXTemplateStrings, (txStr) => {
    let key = '';

    if (txStr.params.key) {
      key = txStr.params.key;
    }

    const partial = createPayload(txStr.string, txStr.params, relativeFile, options);
    if (!isPayloadValid(partial, options)) return;

    mergePayload(HASHES, {
      [key || partial.key]: {
        string: partial.string,
        meta: partial.meta,
      },
    });
  });
}

module.exports = {
  parseHTMLTemplateFile,
};
