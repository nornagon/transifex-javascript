/* globals describe, it, beforeEach, afterEach */

import { expect } from 'chai';
import nock from 'nock';
import { tx, t, generateKey } from '../src/index';

describe('tx instance', () => {
  let fetchTimeout;
  let fetchInterval;

  beforeEach(() => {
    fetchTimeout = tx.fetchTimeout;
    fetchInterval = tx.fetchInterval;
    tx.init({ fetchTimeout: 0, fetchInterval: 0 });
  });

  afterEach(() => {
    nock.cleanAll();
    tx.init({ fetchTimeout, fetchInterval });
  });

  it('getLocales fetches locales', async () => {
    tx.init({
      token: 'abcd',
    });

    nock(tx.cdsHost)
      .get('/languages')
      .reply(200, {
        data: [
          {
            name: 'Greek',
            code: 'el',
            localized_name: 'Ελληνικά',
            rtl: false,
          },
        ],
      });

    const locales = await tx.getLocales({ refresh: true });
    expect(locales).to.deep.equal(['el']);
  });

  it('getLanguages fetches languages', async () => {
    tx.init({
      token: 'abcd',
    });

    nock(tx.cdsHost)
      .get('/languages')
      .reply(200, {
        data: [
          {
            name: 'German',
            code: 'de',
            localized_name: 'German',
            rtl: false,
          },
        ],
      });

    const langs = await tx.getLanguages({ refresh: true });
    expect(langs).to.deep.equal([{
      name: 'German',
      code: 'de',
      localized_name: 'German',
      rtl: false,
    }]);
  });

  it('setCurrentLocale translates strings', async () => {
    tx.init({
      token: 'abcd',
    });

    nock(tx.cdsHost)
      .get('/content/el_GR')
      .reply(200, {
        data: {
          [generateKey('Hello')]: {
            string: 'Γειά',
          },
          [generateKey('World')]: {},
        },
      });

    await tx.setCurrentLocale('el_GR');
    expect(tx.getCurrentLocale()).to.equal('el_GR');
    expect(t('Hello')).to.deep.equal('Γειά');
    expect(t('World')).to.deep.equal('World');

    // restore to source
    await tx.setCurrentLocale('');
    expect(t('Hello')).to.deep.equal('Hello');
  });

  it('setCurrentLocale throws when remote translations are unavailable', async () => {
    tx.init({
      token: 'abcd',
    });

    nock(tx.cdsHost)
      .get('/content/el_GR2')
      .reply(500);

    let threw = false;
    try {
      await tx.setCurrentLocale('el_GR2');
    } catch (err) {
      threw = true;
    }
    expect(threw).to.equal(true);
    expect(tx.getCurrentLocale()).to.equal('');
  });

  it('setCurrentLocale throws when remote translations are invalid', async () => {
    tx.init({
      token: 'abcd',
    });

    nock(tx.cdsHost)
      .get('/content/el_GR2')
      .reply(200, {});

    let threw = false;
    try {
      await tx.setCurrentLocale('el_GR2');
    } catch (err) {
      threw = true;
    }
    expect(threw).to.equal(true);
    expect(tx.getCurrentLocale()).to.equal('');
  });

  it('setCurrentLocale skips when locale is already set', async () => {
    const current = tx.getCurrentLocale();
    await tx.setCurrentLocale(current);
    expect(tx.getCurrentLocale()).to.equal(current);
  });

  it('getLocales throws when remote does not respond', async () => {
    tx.init({
      token: 'abcd',
    });

    nock(tx.cdsHost)
      .get('/languages')
      .reply(500);

    let threw = false;
    try {
      await tx.getLocales({ refresh: true });
    } catch (err) {
      threw = true;
    }
    expect(threw).to.equal(true);
  });

  it('getLocales throws when remote response is wrong', async () => {
    tx.init({
      token: 'abcd',
    });

    nock(tx.cdsHost)
      .get('/languages')
      .reply(200, {});

    let threw = false;
    try {
      await tx.getLocales({ refresh: true });
    } catch (err) {
      threw = true;
    }
    expect(threw).to.equal(true);
  });

  it('getLocales returns empty array when token is not set', async () => {
    tx.init({
      token: '',
    });
    const locales = await tx.getLocales({ refresh: true });
    expect(locales).to.deep.equal([]);
  });

  it('fetchTranslations does not refresh when cache has content', async () => {
    tx.cache.update('el_CACHED', { foo: 'bar' });
    await tx.fetchTranslations('el_CACHED');
    expect(tx.cache.getTranslations('el_CACHED')).to.deep.equal({
      foo: 'bar',
    });
  });

  it('fetchTranslations respects filterTags', async () => {
    const scope = nock(tx.cdsHost)
      .get('/content/lang?filter[tags]=tag1,tag2')
      .reply(200, {
        data: {},
      });

    tx.init({
      token: '',
      filterTags: 'tag1,tag2',
    });
    await tx.fetchTranslations('lang');
    expect(scope.isDone()).to.equal(true);

    // clean up
    tx.init({
      token: '',
      filterTags: null,
    });
  });

  it('retries fetching languages', async () => {
    tx.init({ token: 'abcd' });
    nock(tx.cdsHost)
      .get('/languages')
      .twice()
      .reply(202)
      .get('/languages')
      .reply(200, {
        data: [{
          name: 'Greek',
          code: 'el',
          localized_name: 'Ελληνικά',
          rtl: false,
        }],
      });
    const locales = await tx.getLocales({ refresh: true });
    expect(locales).to.deep.equal(['el']);
  });

  it('retries fetching languages with timeout', async () => {
    tx.init({ token: 'abcd', fetchTimeout: 50 });
    nock(tx.cdsHost)
      .get('/languages')
      .delayConnection(60)
      .reply(202)
      .get('/languages')
      .reply(200, {
        data: [{
          name: 'Greek',
          code: 'el',
          localized_name: 'Ελληνικά',
          rtl: false,
        }],
      });

    let error;
    try {
      await tx.getLocales({ refresh: true });
    } catch (err) {
      error = err;
    }
    expect(error.message).to.equal('Get locales timeout');
  });

  it('retries fetching languages with interval', async () => {
    tx.init({ token: 'abcd', fetchInterval: 50 });
    nock(tx.cdsHost)
      .get('/languages')
      .reply(202)
      .get('/languages')
      .reply(200, {
        data: [{
          name: 'Greek',
          code: 'el',
          localized_name: 'Ελληνικά',
          rtl: false,
        }],
      });

    const ts = Date.now();
    await tx.getLocales({ refresh: true });
    expect(Date.now() - ts).to.be.greaterThan(50);
  });

  it('retries fetching translations', async () => {
    tx.init({ token: 'abcd' });
    nock(tx.cdsHost)
      .get('/content/el')
      .twice()
      .reply(202)
      .get('/content/el')
      .reply(200, { data: { source: { string: 'translation' } } });
    await tx.fetchTranslations('el');
    expect(tx.cache.get('source', 'el')).to.equal('translation');
  });

  it('retries fetching translations with timeout', async () => {
    tx.init({ token: 'abcd', fetchTimeout: 50 });
    nock(tx.cdsHost)
      .get('/content/el_timeout')
      .delayConnection(60)
      .reply(202)
      .get('/content/el_timeout')
      .reply(200, { data: { source: { string: 'translation' } } });

    let error;
    try {
      await tx.fetchTranslations('el_timeout');
    } catch (err) {
      error = err;
    }
    expect(error.message).to.equal('Fetch translations timeout');
  });

  it('retries fetching translations with interval delays', async () => {
    tx.init({ token: 'abcd', fetchInterval: 50 });
    nock(tx.cdsHost)
      .get('/content/el_interval')
      .reply(202)
      .get('/content/el_interval')
      .reply(200, { data: { source: { string: 'translation' } } });

    const ts = Date.now();
    await tx.fetchTranslations('el_interval');
    expect(Date.now() - ts).to.be.greaterThan(50);
  });
});
